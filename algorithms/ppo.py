import torch as th
import gym
import numpy as np
import copy

from torch.distributions import MultivariateNormal

from algorithms.utils.networks import MLP
from algorithms.utils.torch import change_optim_lr
from algorithms.utils.gae import gae


class PPO:
    def __init__(self, env: gym.Env, device="cpu"):
        self.env = env
        self.device = device

        self.action_std = th.diag(1.0, env.action_space.shape[0]).to(device)
        # self.action_std.require_grad = True

        self.actor = MLP(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        self.critic = MLP(env.observation_space.shape[0], 1).to(device)
        self.actor_old = copy.deepcopy(self.actor)
        self.actor_old.eval()
        self.buffer = []

        self.optim = th.optim.Adam(params=[{"actor": self.actor.parameters()},
                                           {"critic": self.critic.parameters()}], lr=1e-4)

    def rollout(self, rollout_steps):
        obs1 = self.env.reset()

        for _ in range(rollout_steps):
            action = self.predict(obs1)
            obs2, reward, done, info = self.env.step(action)
            self.buffer.append([obs1, action, reward, done, obs2])
            obs1 = obs2

            if done:
                obs1 = self.env.reset()

    def update(self, num_epochs, lam, gamma, eps, beta, value_coef):
        # include both actor and critic loss
        loss = self.compute_loss(num_epochs, lam, gamma, eps, beta, value_coef)

        self.optim.zero_grad()
        loss.loss.backward()
        self.optim.step()

        # Copy new weights into old policy
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.buffer = []

    def learn(self, total_step, rollout_step,
              num_epochs, lam, gamma, eps, beta, value_coef, lr=1e-4):
        change_optim_lr(self.optim, lr)

        for _ in range(total_step // rollout_step):
            self.rollout(rollout_step)
            self.update(num_epochs, lam, gamma, eps, beta, value_coef)

    def predict(self, obs, deterministic=False) -> np.ndarray:
        self.actor.eval()
        with th.no_grad():
            if deterministic:
                action = self.actor(obs).detach().cpu().numpy()
                self.actor.train()
            else:
                distr = MultivariateNormal(self.actor(obs), self.action_std)
                action = distr.sample().detach().cpu().numpy()
                self.actor.train()

        return action

    def compute_loss(self, num_epochs, lam, gamma, eps, beta, value_coef):
        states, actions, rewards, old_logprobs = self.buffer

        old_logprobs, _, _ = self.evaluate_policy(states, actions)
        # Optimize policy for num_epochs
        loss = None
        for _ in range(num_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.evaluate_policy(states, actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = th.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = gae(t=0, states=states, rewards=rewards, value_net=self.critic, lam=lam, gamma=gamma)
            surr1 = ratios * advantages
            surr2 = th.clamp(ratios, 1 - eps, 1 + eps) * advantages
            loss = -th.min(surr1, surr2) + value_coef * th.nn.MSELoss()(state_values, rewards) - beta * dist_entropy

        return loss

    def evaluate_policy(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_std.expand_as(action_mean)
        cov_mat = th.diag_embed(action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, th.squeeze(state_value), dist_entropy
