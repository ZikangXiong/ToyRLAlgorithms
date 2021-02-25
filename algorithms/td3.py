import os
import copy

import torch as th
import gym
from algorithms.utils.torch import polyak_update, grad_clip

from algorithms.ddpg import DDPG
from algorithms.utils.networks import MLP


class TD3(DDPG):

    def __init__(self, env: gym.Env, device, buffer_size=1000000):
        super(TD3, self).__init__(env, device, buffer_size)
        self.critic2 = MLP(env.observation_space.shape[0] + env.action_space.shape[0], 1).to(device)
        self.critic2_optim = th.optim.Adam(params=self.actor.parameters(), lr=1e-4)
        self.critic2_target = copy.deepcopy(self.critic)
        self.critic2_target.eval()
        self.target_action_noise = None

    def update_critic(self, training_batch, gamma, grad_clip_val):
        critic_loss, critic2_loss = self.compute_critic_loss(training_batch, gamma)

        self.critic_optim.zero_grad()
        grad_clip(self.critic, grad_clip_val)
        critic_loss.backward()
        self.critic_optim.step()

        self.critic2_optim.zero_grad()
        grad_clip(self.critic2, grad_clip_val)
        critic2_loss.backward()
        self.critic2_optim.step()

    def compute_target_q(self, sample_batch, gamma):
        # compute target Q
        with th.no_grad():
            s2 = th.from_numpy(sample_batch[4]).float().to(self.device)
            action_pred = self.actor_target(s2)
            action_noise = 0 if self.target_action_noise is None else self.target_action_noise * (
                        th.rand(action_pred.shape) - 0.5)
            target_critic_input = th.cat([s2, action_pred + action_noise], dim=1)
            q_pred1 = self.critic_target(target_critic_input)
            q_pred2 = self.critic2_target(target_critic_input)
            reward = th.from_numpy(sample_batch[2]).float().to(self.device)
            dones = sample_batch[3]
            q_pred = th.min(th.cat([q_pred1, q_pred2], dim=1), dim=1)[0].unsqueeze(-1)
            target_q = reward + (gamma * q_pred * (1 - dones))

        return target_q

    def compute_critic_loss(self, sample_batch, gamma):
        target_q = self.compute_target_q(sample_batch, gamma)

        s1 = th.from_numpy(sample_batch[0]).float().to(self.device)
        actions = th.from_numpy(sample_batch[1]).float().to(self.device)
        critic_input = th.cat([s1, actions], dim=1)
        q1 = self.critic(critic_input)
        q2 = self.critic2(critic_input)

        loss_fn = th.nn.MSELoss()
        loss1 = loss_fn(q1, target_q)
        loss2 = loss_fn(q2, target_q)

        return loss1, loss2

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
        super(TD3, self).learn(learning_steps,
                               rollout_steps,
                               actor_lr,
                               critic_lr,
                               batch_size,
                               n_epoch,
                               n_step,
                               gamma, tau,
                               grad_clip_val,
                               noise_param)

        self.target_action_noise = kwargs.get("target_action_noise", None)

    def update_target(self, tau):
        polyak_update(self.actor, self.actor_target, tau)
        polyak_update(self.critic, self.critic_target, tau)
        polyak_update(self.critic2, self.critic2_target, tau)

    def save(self, file_path):
        save_dict = {
            "device": self.device,
            "buffer_size": self.replay_buffer.size(),
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
        }

        if not os.path.isdir(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        th.save(save_dict, file_path)

    @classmethod
    def load(cls, file_path, env=None):
        save_dict = th.load(file_path)
        td3 = cls(env, device=save_dict["device"], buffer_size=save_dict["buffer_size"])
        td3.actor.load_state_dict(save_dict["actor"])
        td3.actor_target.load_state_dict(save_dict["actor_target"])
        td3.critic.load_state_dict(save_dict["critic"])
        td3.critic_target.load_state_dict(save_dict["critic_target"])
        td3.critic2.load_state_dict(save_dict["critic2"])
        td3.critic2_target.load_state_dict(save_dict["critic2_target"])

        return td3
