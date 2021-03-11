from typing import Tuple

import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn

from auto_rl.learning.distribution import create_actor_distribution
from auto_rl.learning.network import MLP
from auto_rl.utils.gym import infer_action_type


class MLPActorCritic(nn.Module):
    def __init__(self, env, arch=None, device="cpu"):
        super(MLPActorCritic, self).__init__()

        self.env = env
        self.state_size = self.env.observation_space.high.shape[0]
        self.action_size = self.env.action_space.high.shape[0]
        self.action_type = infer_action_type(self.env)
        self.arch = arch

        if self.action_type == "continuous":
            self.actor_encoder, self.critic_encoder, self.action_decoder, self.value_decoder = self._build_network()
            self.actor, self.critic = self._build_actor_critic()
        else:
            raise NotImplementedError()

        self.device = device

    def forward(self, x):
        if self.actor_encoder is self.critic_encoder:
            actor_embedding = self.actor_encoder(x)
            critic_embedding = actor_embedding
        else:
            actor_embedding = self.actor_encoder(x)
            critic_embedding = self.critic_encoder(x)

        action_mean_var = self.action_decoder(actor_embedding)
        value = self.value_decoder(critic_embedding)

        return action_mean_var, value, actor_embedding, critic_embedding

    def compute_actor_embedding(self, obs: np.ndarray) -> np.ndarray:
        one_dim = len(obs.shape) == 1

        self.eval()
        with th.no_grad():
            obs = self._preprocess_numpy_to_tensor(obs)
            embedding = self.actor_encoder(obs)
        self.actor.train()

        action = embedding.detach().cpu().numpy()
        if one_dim:
            embedding = action.reshape(-1)

        return embedding

    def predict(self, obs: np.ndarray, deterministic=False) -> np.ndarray:
        one_dim = len(obs.shape) == 1

        self.eval()
        with th.no_grad():
            obs = self._preprocess_numpy_to_tensor(obs)
            mean_var = self.actor(obs)

            if deterministic:
                action = mean_var[:, :self.action_size]
            else:
                distribution = create_actor_distribution(self.action_type, mean_var, self.action_size)
                action = distribution.sample()
        self.train()

        action = action.detach().cpu().numpy()
        if one_dim:
            action = action.reshape(-1)

        # scale if last layer is tanh
        if type(self.action_decoder.layers[-1]) is th.nn.Tanh:
            action *= self.env.action_space.high

        return action

    def eval_policy(self, states: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        action_mean_var, value, _, _ = self.forward(states)

        # unscale if last layer is tanh
        if type(self.action_decoder.layers[-1]) is th.nn.Tanh:
            actions /= th.tensor(self.env.action_space.high).float().to(self.device)

        distribution = create_actor_distribution(self.action_type, action_mean_var, self.action_size)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return value.squeeze(), log_prob.sum(dim=1), entropy.sum(dim=1)

    def _build_network(self):
        if self.arch is None:
            self.arch = {
                "share": [64, 32],
            }

            # self.arch = {
            #     "actor": [64, 32],
            #     "critic": [64, 32]
            # }

        if self.arch.get("share", None) is not None:
            actor_arch_list = self.arch["share"]
            critic_arch_list = self.arch["share"]
            actor_encoder = self._build_encoder(actor_arch_list)
            critic_encoder = actor_encoder
        else:
            actor_arch_list = self.arch["actor"]
            critic_arch_list = self.arch["critic"]
            actor_encoder = self._build_encoder(actor_arch_list)
            critic_encoder = self._build_encoder(critic_arch_list)

        # decode mean and variance of action
        action_decoder = MLP([nn.Linear(actor_arch_list[-1], self.action_size * 2), th.nn.Tanh()])
        value_decoder = nn.Linear(critic_arch_list[-1], 1)

        return actor_encoder, critic_encoder, action_decoder, value_decoder

    def _build_actor_critic(self):
        actor = MLP([self.actor_encoder, self.action_decoder])
        critic = MLP([self.critic_encoder, self.value_decoder])

        return actor, critic

    def _build_encoder(self, arch_list):
        encoder_layers = []
        prev_layer_neuron_num = self.state_size

        for i, neuron_num in enumerate(arch_list):
            fc = nn.Linear(prev_layer_neuron_num, neuron_num)
            prev_layer_neuron_num = neuron_num
            encoder_layers.append(fc)
            if i < len(arch_list) - 1:
                encoder_layers.append(F.relu)

        return MLP(encoder_layers)

    def _preprocess_numpy_to_tensor(self, obs: np.ndarray) -> th.Tensor:
        if len(obs.shape) == 1:
            obs = obs.reshape((1, len(obs)))
        obs = th.from_numpy(obs).float().to(self.device)

        return obs
