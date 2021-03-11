import os
import shutil

import gym

from unittest import TestCase
from auto_rl.learning.ppo import PPO
from auto_rl.learning.policy import MLPActorCritic


class Test(TestCase):
    def test_log_actor_critic_graph(self):
        device = "cpu"
        env = gym.make("Pendulum-v0")
        policy = MLPActorCritic(env)
        if os.path.isdir("./log"):
            shutil.rmtree("./log")
        PPO(env, policy, device, "./log")
        # manual check
        "tensorboard --logdir=auto_rl/tests/utils"
