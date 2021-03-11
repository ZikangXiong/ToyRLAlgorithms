from unittest import TestCase
import gym
import numpy as np
import copy

from auto_rl.learning.ppo import PPO
from auto_rl.learning.policy import MLPActorCritic
from auto_rl.utils.torch import validate_state_dicts


class TestPPO(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPPO, self).__init__(*args, **kwargs)

        self.device = "cpu"
        self.env = gym.make("Pendulum-v0")
        self.policy = MLPActorCritic(self.env)

    def test_predict_shape(self):
        ppo = PPO(self.env, self.policy, self.device)
        obs = self.env.observation_space.sample()
        action = ppo.predict(obs, deterministic=False)
        self.assertTrue(action.shape == (1,))
        action = ppo.predict(np.array([obs, obs]), deterministic=False)
        self.assertTrue(action.shape == (2, 1))

    def test_predict_deterministic(self):
        ppo = PPO(self.env, self.policy, self.device)
        obs = self.env.observation_space.sample()
        action1 = ppo.predict(obs, deterministic=True)
        action2 = ppo.predict(obs, deterministic=True)
        self.assertTrue(action1 == action2)

    def test_rollout_buffer(self):
        ppo = PPO(self.env, self.policy, self.device)
        ppo.rollout(500)
        for i in range(5):
            self.assertTrue(len(ppo.buffer.buffer[i]) == 500)

    def test_update_lr(self):
        ppo = PPO(self.env, self.policy, self.device)
        ppo.update(lr=1e-3,
                   optimize_epoch_num=0,
                   batch_size=-1,
                   grad_clip_cnst=0.1,
                   gamma=0.99,
                   entropy_coef=0.05,
                   lam=0.5,
                   ratio_clip_cnst=0.1,
                   value_coef=0.5)
        for param_group in ppo.optimizer.param_groups:
            self.assertTrue(param_group["lr"] == 1e-3)

    def test_update_policy_change(self):
        ppo = PPO(self.env, self.policy, self.device)

        ori_policy = copy.deepcopy(self.policy)
        self.assertTrue(validate_state_dicts(ppo.policy.actor.state_dict(), ori_policy.actor.state_dict()))
        self.assertTrue(validate_state_dicts(ppo.policy.critic.state_dict(), ori_policy.critic.state_dict()))

        ppo.rollout(200)
        ppo.update(lr=1e-3,
                   optimize_epoch_num=1,
                   batch_size=-1,
                   grad_clip_cnst=0.1,
                   gamma=0.99,
                   entropy_coef=0.05,
                   gae_lam=None,
                   ratio_clip_cnst=0.1,
                   value_coef=0.5)

        self.assertFalse(validate_state_dicts(ppo.policy.actor.state_dict(), ori_policy.actor.state_dict()))
        self.assertFalse(validate_state_dicts(ppo.policy.critic.state_dict(), ori_policy.critic.state_dict()))

    def test_update_old_policy_change(self):
        ppo = PPO(self.env, self.policy, self.device)
        ori_policy = copy.deepcopy(ppo.policy_old)

        ppo.rollout(200)
        ppo.update(lr=1e-3,
                   optimize_epoch_num=1,
                   batch_size=-1,
                   grad_clip_cnst=0.1,
                   gamma=0.99,
                   entropy_coef=0.05,
                   gae_lam=None,
                   ratio_clip_cnst=0.1,
                   value_coef=0.5)

        self.assertFalse(validate_state_dicts(ppo.policy_old.actor.state_dict(), ori_policy.actor.state_dict()))
        self.assertFalse(validate_state_dicts(ppo.policy_old.critic.state_dict(), ori_policy.critic.state_dict()))

    def test_update_buffer_empty(self):
        ppo = PPO(self.env, self.policy, self.device)
        ppo.rollout(200)
        ppo.update(lr=2.5e-4,
                   optimize_epoch_num=1,
                   batch_size=-1,
                   grad_clip_cnst=0.1,
                   gamma=0.99,
                   entropy_coef=0.05,
                   gae_lam=None,
                   ratio_clip_cnst=0.2,
                   value_coef=0.5)
        for item in ppo.buffer.buffer:
            self.assertTrue(item == [])

    def test_optimizer_register_right_params(self):
        ppo = PPO(self.env, self.policy, self.device)
        self.assertTrue(len(ppo.optimizer.param_groups[0]["params"]) == 10)

    def test_integration(self):
        ppo = PPO(self.env, self.policy, self.device)

        ppo.learn(total_steps=2000000,
                  rollout_steps=4000,
                  lr=3e-4,
                  optimize_epoch_num=50,
                  batch_size=None,
                  gamma=0.99,
                  gae_lam=None,
                  ratio_clip_cnst=0.2,
                  entropy_coef=0.01,
                  value_coef=0.5,
                  grad_clip_cnst=None,
                  eval_intv=None)
