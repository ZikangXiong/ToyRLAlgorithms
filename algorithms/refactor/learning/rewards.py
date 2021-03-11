from typing import Tuple

import numpy as np
import torch as th


class GAE:

    def __init__(self, n_workers: int, worker_steps: int, gamma: float, lambda_: float, device: str):
        """
        https://nn.labml.ai/rl/ppo/gae.html
        :param n_workers:
        :param worker_steps:
        :param gamma:
        :param lambda_:
        """
        self.lambda_ = lambda_
        self.gamma = gamma
        self.worker_steps = worker_steps
        self.n_workers = n_workers
        self.device = device

    def __call__(self, done: th.Tensor, rewards: th.Tensor, values: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        advantages = th.zeros(self.n_workers, self.worker_steps).float().to(self.device)
        last_advantage = 0
        last_value = values[:, -1]
        values_ests = []

        for t in reversed(range(self.worker_steps)):
            mask = th.logical_not(done[:, t])
            last_value = last_value * mask
            last_advantage = last_advantage * mask

            values_ests.append(rewards[:, t] + self.gamma * last_value)
            delta = values_ests[-1] - values[:, t]
            last_advantage = delta + self.gamma * self.lambda_ * last_advantage
            advantages[:, t] = last_advantage
            last_value = values[:, t]

        values_ests = th.tensor(values_ests).float().to(done.device)

        return advantages, values_ests


def single_worker_gae(values: th.Tensor,
                      dones: th.Tensor,
                      rewards: th.Tensor,
                      gamma: float, lam: float, device: str) -> Tuple[th.Tensor, th.Tensor]:
    gae = GAE(1, len(values), gamma, lam, device)

    values = values.view((1,) + values.shape)
    dones = dones.view((1,) + dones.shape)
    rewards = rewards.view((1,) + rewards.shape)
    ret = gae(dones, rewards, values)
    return ret[0][0], ret[1][0]


def mc_reward_estimation(rewards, dones, gamma) -> np.ndarray:
    value_est = []
    discounted_reward = 0

    for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + (gamma * discounted_reward)
        value_est.insert(0, discounted_reward)

    return np.array(value_est)
