from typing import Tuple
import os

import torch as th
import numpy as np
import gym
import copy

from torch.utils.tensorboard import SummaryWriter

from auto_rl.learning.policy import MLPActorCritic
from auto_rl.learning.buffer import OnPolicyBuffer
from auto_rl.learning.rewards import single_worker_gae, mc_reward_estimation
from auto_rl.utils.torch import change_optim_lr, grad_clip
from auto_rl.utils.gym import infer_action_size, infer_action_type
from auto_rl.simulation.run import single_worker_rollout, rollout_rew, eval_with_render
from auto_rl.utils.tensorboard import log_actor_critic_graph
from auto_rl.utils.logger import Logger


class PPO:
    def __init__(self,
                 env: gym.Env,
                 policy: MLPActorCritic,
                 device: str,
                 log_dir=None):
        assert policy.device == device

        self.env = env
        self.policy = policy
        self.device = device

        # general logger
        self.logger = Logger()

        # Tensorboard writer
        self.enable_tensorboard = False
        if log_dir is not None:
            self.enable_tensorboard = True

        if self.enable_tensorboard:
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            self.tb_writer = SummaryWriter(log_dir)
            # log computational graph
            log_actor_critic_graph(self.tb_writer, self.env, self.policy, self.device)

        # Initialize optimizer
        self.optimizer = th.optim.Adam(params=self.policy.parameters(), lr=1e-4)

        # Old policy
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.eval()

        self.mse_loss = th.nn.MSELoss()
        self.buffer = OnPolicyBuffer()

        # Action type and size
        self.action_type = infer_action_type(self.env)
        self.action_size = infer_action_size(self.env, self.action_type)

    def predict(self, obs: np.ndarray, deterministic=False):
        action = self.policy_old.predict(obs, deterministic)

        return action

    def rollout(self, rollout_steps):
        single_worker_rollout(self.env, self.policy, self.buffer, rollout_steps)

        # Log
        rew_mean, rew_min, rew_max = rollout_rew(self.buffer)
        self.logger.add("rew. avg.", rew_mean)
        self.logger.add("rew. min", rew_min)
        self.logger.add("rew. max", rew_max)

        if self.enable_tensorboard:
            self.tb_writer.add_scalar("reward/mean", rew_mean)
            self.tb_writer.add_scalar("reward/min", rew_min)
            self.tb_writer.add_scalar("reward/max", rew_max)

    def update(self, lr, optimize_epoch_num, batch_size,
               gamma, gae_lam, ratio_clip_cnst,
               entropy_coef, value_coef, grad_clip_cnst):
        change_optim_lr(self.optimizer, lr)

        loss, value, entropy = None, None, None
        for _ in range(optimize_epoch_num):
            loss, value, entropy = self.compute_loss(batch_size, gamma, gae_lam,
                                                     ratio_clip_cnst, entropy_coef, value_coef)

            self.optimizer.zero_grad()
            loss.backward()
            if grad_clip_cnst is not None:
                grad_clip(self.policy, grad_clip_cnst)
            self.optimizer.step()

        # Log
        if loss is not None:
            self.logger.add("loss", loss.detach().cpu().numpy())
            self.logger.add("value", value.detach().cpu().numpy())
            self.logger.add("entropy", entropy.detach().cpu().numpy())
            if self.enable_tensorboard:
                self.tb_writer.add_scalar("loss/loss", loss)
                self.tb_writer.add_scalar("loss/value", value)
                self.tb_writer.add_scalar("loss/entropy", entropy)
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        assert not self.policy_old.training
        self.buffer.clear()

    def learn(self, total_steps,
              rollout_steps,
              lr,
              optimize_epoch_num,
              batch_size,
              gamma,
              gae_lam,
              ratio_clip_cnst,
              entropy_coef,
              value_coef,
              grad_clip_cnst=None,
              eval_intv=None):

        for i in range(total_steps // rollout_steps + 1):
            self.rollout(rollout_steps)
            self.update(lr, optimize_epoch_num, batch_size,
                        gamma, gae_lam, ratio_clip_cnst,
                        entropy_coef, value_coef,
                        grad_clip_cnst)

            # Log output
            self.logger.dump()

            # evaluate with video
            if eval_intv is not None and i % eval_intv == 0:
                eval_with_render(self.env, self.policy)

    def compute_loss(self, batch_size, gamma, gae_lam,
                     ratio_clip_cnst,
                     entropy_coef, value_coef, use_gae=False) \
            -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        if batch_size is None:
            # read all data, no batch
            s1, actions, rewards, dones, s2 = self.buffer.read()
        else:
            s1, actions, rewards, dones, s2 = self.buffer.sample(batch_size)
            assert not use_gae, "Inefficient to compute GAE from random sample."

        s1 = th.from_numpy(s1).float().to(self.device)
        actions = th.from_numpy(actions).float().to(self.device)

        _, old_log_probs, _ = self.policy_old.eval_policy(s1, actions)
        assert self.policy.training
        values, log_probs, entropy = self.policy.eval_policy(s1, actions)

        advantages, value_estimation = self.compute_advantage(gae_lam, dones, rewards, values, gamma)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        value_estimation = (value_estimation - value_estimation.mean()) / (value_estimation.std() + 1e-8)

        ratios = th.exp(log_probs - old_log_probs.detach())
        surr1 = ratios * advantages
        surr2 = th.clamp(ratios, 1 - ratio_clip_cnst, 1 + ratio_clip_cnst) * advantages

        loss = -th.min(surr1, surr2).mean() - entropy_coef * entropy.mean()
        loss = loss + value_coef * self.mse_loss(values, value_estimation)

        return loss, values.mean(), entropy.mean()

    def compute_advantage(self, gae_lam, dones, rewards, values, gamma):
        # FIXME: Understand GAE fully and write this part
        """if gae_lam is not None:
            dones = th.from_numpy(dones).float().to(self.device)
            rewards = th.from_numpy(rewards).float().to(self.device)
            advantages, value_estimation = single_worker_gae(values, dones, rewards, gamma, gae_lam, self.device)
        else:"""
        value_estimation = mc_reward_estimation(rewards, dones, gamma)
        value_estimation = th.tensor(value_estimation).float().to(self.device)
        advantages = value_estimation - values.detach()

        return advantages, value_estimation

    def __del__(self):
        if self.enable_tensorboard:
            self.tb_writer.close()
