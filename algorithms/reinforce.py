import os

import torch as th
import numpy as np
import gym

from torch.distributions import MultivariateNormal

from algorithms.utils.networks import MLP
from algorithms.utils.torch import change_optim_lr


class Reinforce:
    # On-policy version
    def __init__(self, env: gym.Env, device="cpu"):
        self.env = env
        self.buffer = []
        self.policy_network = MLP(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        # Support continuous variables
        self.action_std = th.diag(1.0, env.action_space.shape[0]).to(device)
        self.action_std.require_grad = True
        self.device = device

        self.optim = th.optim.Adam(self.policy_network.parameters(), lr=1e-4)

    def rollout(self, rollout_steps):
        obs1 = self.env.reset()

        for _ in range(rollout_steps):
            action = self.predict(obs1)
            obs2, reward, done, info = self.env.step(action)
            self.buffer.append([obs1, action, reward, done, obs2])
            obs1 = obs2

            if done:
                obs1 = self.env.reset()

    def update(self, gamma):
        loss = self.policy_grad(gamma)

        # On policy change, without important sampling
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.buffer = []

    def learn(self, total_step, rollout_step, lr=1e-4, gamma=0.99):
        change_optim_lr(self.optim, lr)

        for _ in range(total_step // rollout_step):
            self.rollout(rollout_step)
            self.update(gamma)

    def predict(self, obs, deterministic=False) -> np.ndarray:
        self.policy_network.eval()
        with th.no_grad():
            if deterministic:
                action = self.policy_network(obs).detach().cpu().numpy()
                self.policy_network.train()
            else:
                distr = MultivariateNormal(self.policy_network(obs), self.action_std)
                action = distr.sample().detach().cpu().numpy()
                self.policy_network.train()

        return action

    def policy_grad(self, gamma) -> th.Tensor:
        dis_rews = self.compute_discounted_rewards(gamma)
        log_probs = self.compute_log_probs()

        return dis_rews.dot(log_probs)

    def compute_discounted_rewards(self, gamma):
        buffer = np.array(self.buffer)
        r = th.from_numpy(buffer[2:]).float().to(self.device)
        dones = buffer[3, :]
        dis_rew = r * (gamma ** np.arange(len(r)))

        return self.group_sum_with_done(dis_rew, dones)

    def compute_log_probs(self):
        buffer = np.array(self.buffer)
        obs = buffer[0, :]
        action = buffer[1, :]
        dones = buffer[3, :]

        distr = MultivariateNormal(self.policy_network(obs), self.action_std)
        log_prob = distr.log_prob(action)

        return self.group_sum_with_done(log_prob, dones)

    def group_sum_with_done(self, inp, dones):
        ind = np.where(dones) + 1

        ret = [th.sum(inp[:ind[0]])]
        for i in range(1, len(ind) + 1):
            ret.append(th.sum(inp[i - 1:i]))

        return th.tensor(ret, device=self.device)

    def save(self, file_path):
        save_dict = {
            "net": self.policy_network.state_dict(),
            "action_std": self.action_std,
            "device": self.device
        }
        if not os.path.isdir(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        th.save(save_dict, file_path)

    @classmethod
    def load(cls, file_path, env):
        save_dict = th.load(file_path)
        agent = cls(env, save_dict["device"])
        agent.policy_network.load_state_dict(save_dict["net"])
        agent.action_std = save_dict["action_std"]

        return agent
