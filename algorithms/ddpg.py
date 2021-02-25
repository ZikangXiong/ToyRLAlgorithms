import os
import torch as th
import numpy as np
import gym

from algorithms.utils.networks import MLP
from algorithms.utils.replay_buffer import ReplayBuffer
from algorithms.utils.torch import change_optim_lr, grad_clip, polyak_update
from algorithms.utils.noise import OrnsteinUhlenbeckActionNoise

import copy


class DDPG:
    """
    Actor is the argmax in DQN
    """

    def __init__(self, env: gym.Env, device, buffer_size=1000000):
        self.env = env
        self.device = device

        # Did not share the encoder
        self.actor = MLP(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        self.critic = MLP(env.observation_space.shape[0] + env.action_space.shape[0], 1).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_target.eval()
        self.critic_target.eval()

        self.actor_optim = th.optim.Adam(params=self.actor.parameters(), lr=1e-4)
        self.critic_optim = th.optim.Adam(params=self.critic.parameters(), lr=1e-4)

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size)

        self.noise = None

    def rollout(self, rollout_step):
        obs1 = self.env.reset()
        self.noise.reset()

        for _ in range(rollout_step):
            action = self.predict(obs1)
            obs2, reward, done, info = self.env.step(action)
            self.replay_buffer.add(obs1, action, reward, done, obs2)
            obs1 = obs2

            if done:
                obs1 = self.env.reset()
                self.noise.reset()

    def update(self, actor_lr, critic_lr, batch_size, n_epoch, n_step, gamma, tau, grad_clip_val):
        change_optim_lr(self.actor_optim, actor_lr)
        change_optim_lr(self.critic_optim, critic_lr)

        for _ in range(n_epoch):
            for _ in range(n_step):
                training_batch = self.replay_buffer.sample_batch(batch_size)
                self.update_critic(training_batch, gamma, grad_clip_val)
                self.update_actor(training_batch, grad_clip_val)
                self.update_target(tau)

    def update_critic(self, training_batch, gamma, grad_clip_val):
        critic_loss = self.compute_critic_loss(training_batch, gamma)
        self.critic_optim.zero_grad()
        grad_clip(self.critic, grad_clip_val)
        critic_loss.backward()
        self.critic_optim.step()

    def update_actor(self, training_batch, grad_clip_val):
        actor_loss = self.compute_actor_loss(training_batch)
        self.actor_optim.zero_grad()
        grad_clip(self.actor, grad_clip_val)
        actor_loss.backward()
        self.actor_optim.step()

    def update_target(self, tau):
        polyak_update(self.actor, self.actor_target, tau)
        polyak_update(self.critic, self.critic_target, tau)

    def compute_critic_loss(self, sample_batch, gamma):
        target_q = self.compute_target_q(sample_batch, gamma)

        # compute Q
        s1 = th.from_numpy(sample_batch[0]).float().to(self.device)
        actions = th.from_numpy(sample_batch[1]).float().to(self.device)
        critic_input = th.cat([s1, actions], dim=1)
        q = self.critic(critic_input)

        loss_fn = th.nn.MSELoss()
        loss = loss_fn(q, target_q)

        return loss

    def compute_target_q(self, sample_batch, gamma):
        # compute target Q
        with th.no_grad():
            s2 = th.from_numpy(sample_batch[4]).float().to(self.device)
            action_pred = self.actor_target(s2)
            target_critic_input = th.cat([s2, action_pred], dim=1)
            q_pred = self.critic_target(target_critic_input)
            reward = th.from_numpy(sample_batch[2]).float().to(self.device)
            dones = sample_batch[3]
            target_q = reward + (gamma * q_pred * (1 - dones))

        return target_q

    def compute_actor_loss(self, sample_batch):
        s1 = th.from_numpy(sample_batch[0]).float().to(self.device)
        action_pred = self.actor(s1)
        # by update actor network, maximize the critic output
        actor_loss = -self.critic(th.cat((s1, action_pred), dim=1)).mean()

        return actor_loss

    def learn(self, learning_steps,
              rollout_steps,
              actor_lr,
              critic_lr,
              batch_size,
              n_epoch,
              n_step,
              gamma, tau,
              grad_clip_val,
              noise_param,
              **kwargs):

        # The original paper used Ornstein-uhlenbeck process
        self.noise = OrnsteinUhlenbeckActionNoise(**noise_param)

        for _ in range(learning_steps // rollout_steps):
            self.rollout(rollout_steps)
            self.update(actor_lr, critic_lr, batch_size, n_epoch, n_step, gamma, tau, grad_clip_val)

    def predict(self, obs, deterministic=False) -> np.ndarray:
        obs = th.from_numpy(obs).float().to(self.device)

        self.actor.eval()
        with th.no_grad():
            if not deterministic:
                action = self.actor(obs).cpu().numpy() + self.noise()
            else:
                action = self.actor(obs).cpu().numpy()
        self.actor.train()

        return action

    def save(self, file_path):
        save_dict = {
            "device": self.device,
            "buffer_size": self.replay_buffer.size(),
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }

        if not os.path.isdir(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        th.save(save_dict, file_path)

    @classmethod
    def load(cls, file_path, env=None):
        save_dict = th.load(file_path)
        ddpg = cls(env, device=save_dict["device"], buffer_size=save_dict["buffer_size"])
        ddpg.actor.load_state_dict(save_dict["actor"])
        ddpg.actor_target.load_state_dict(save_dict["actor_target"])
        ddpg.critic.load_state_dict(save_dict["critic"])
        ddpg.critic_target.load_state_dict(save_dict["critic_target"])

        return ddpg
