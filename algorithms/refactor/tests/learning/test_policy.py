from unittest import TestCase
from auto_rl.learning.policy import MLPActorCritic
import numpy as np
import torch as th
import gym


class TestMLPActorCritic(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMLPActorCritic, self).__init__(*args, **kwargs)
        self.env = gym.make("Pendulum-v0")

    def test_build_default_share_network_actor(self):
        mlp_ac = MLPActorCritic(self.env)
        self.assertTrue(mlp_ac.actor_encoder.layers[0].in_features == 3)
        self.assertTrue(mlp_ac.action_decoder.layers[-2].out_features == 2)

    def test_build_default_share_network_critic(self):
        mlp_ac = MLPActorCritic(self.env)
        self.assertTrue(mlp_ac.critic_encoder.layers[0].in_features == 3)
        self.assertTrue(mlp_ac.value_decoder.out_features == 1)

    def test_build_default_share_network_share(self):
        mlp_ac = MLPActorCritic(self.env)
        self.assertTrue(mlp_ac.actor.layers[0] == mlp_ac.critic.layers[0])
        self.assertFalse(mlp_ac.actor.layers[1] == mlp_ac.critic.layers[1])

    def test_compute_actor_embedding(self):
        mlp_ac = MLPActorCritic(self.env)
        obs = self.env.observation_space.sample()
        embedding = mlp_ac.compute_actor_embedding(obs)
        self.assertTrue(type(embedding) is np.ndarray)
        self.assertTrue(len(embedding) == mlp_ac.actor_encoder.model[-1].out_features)

    def test_predict_shape(self):
        mlp_ac = MLPActorCritic(self.env)
        obs = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        action = mlp_ac.predict(obs, deterministic=True)
        self.assertTrue(action.shape == (1,))
        action = mlp_ac.predict(obs, deterministic=False)
        self.assertTrue(action.shape == (1,))

    def test_predict_deterministic(self):
        mlp_ac = MLPActorCritic(self.env)
        obs = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        action1 = mlp_ac.predict(obs, deterministic=True)
        action2 = mlp_ac.predict(obs, deterministic=True)
        self.assertTrue((action1 == action2).all())

    def test_predict_stochastic(self):
        mlp_ac = MLPActorCritic(self.env)
        obs = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        action1 = mlp_ac.predict(obs, deterministic=False)
        action2 = mlp_ac.predict(obs, deterministic=False)
        self.assertTrue((action1 != action2).any())

    def test_eval_policy(self):
        mlp_ac = MLPActorCritic(self.env)
        obs1 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        action1 = mlp_ac.predict(obs1, deterministic=False)
        obs2 = np.array([2.0, 1.0, 1.0], dtype=np.float32)
        action2 = mlp_ac.predict(obs2, deterministic=False)

        value, log_prob, entropy = mlp_ac.eval_policy(th.from_numpy(np.array([obs1, obs2])),
                                           th.from_numpy(np.array([action1, action2])))

        self.assertTrue(value.shape == (2,))
        self.assertTrue(log_prob.shape == (2, ))
        self.assertTrue(entropy.shape == (2, ))
