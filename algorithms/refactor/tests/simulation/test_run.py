import gym

from auto_rl.learning.buffer import OnPolicyBuffer
from auto_rl.simulation.run import rollout_rew, eval_with_render
from auto_rl.learning.policy import MLPActorCritic

from unittest import TestCase


class Test(TestCase):

    def test_rollout_rew(self):
        buffer = OnPolicyBuffer()
        self.assertTrue(rollout_rew(buffer) == (None, None, None))

        buffer.add(1, 1, 1.0, False, 2)
        buffer.add(2, 2, 2.0, False, 3)
        buffer.add(3, 3, 3.0, True, 4)
        buffer.add(1, 1, 2.0, False, 2)
        buffer.add(2, 2, 3.0, False, 3)
        buffer.add(3, 3, 4.0, True, 4)
        self.assertTrue(rollout_rew(buffer) == (7.5, 6.0, 9.0))
        buffer.clear()

        buffer.add(1, 1, 1.0, False, 2)
        buffer.add(2, 2, 2.0, False, 3)
        buffer.add(3, 3, 3.0, True, 4)
        self.assertTrue(rollout_rew(buffer) == (6.0, 6.0, 6.0))
        buffer.clear()

        buffer.add(3, 3, 3.0, True, 4)
        self.assertTrue(rollout_rew(buffer) == (3.0, 3.0, 3.0))
        buffer.clear()

        buffer.add(1, 1, 1.0, False, 2)
        buffer.add(2, 2, 2.0, False, 3)
        buffer.add(3, 3, 3.0, False, 4)
        self.assertTrue(rollout_rew(buffer) == (6.0, 6.0, 6.0))
        buffer.clear()

        buffer.add(1, 1, 1.0, False, 2)
        buffer.add(2, 2, 2.0, False, 3)
        buffer.add(3, 3, 3.0, True, 4)
        buffer.add(1, 1, 2.0, False, 1)
        buffer.add(2, 2, 3.0, False, 3)
        buffer.add(3, 3, 4.0, False, 4)
        self.assertTrue(rollout_rew(buffer) == (7.5, 6.0, 9.0))
        buffer.clear()

    def _test_eval_with_render(self):
        policy = MLPActorCritic(3, 1, "continuous")
        env = gym.make("Pendulum-v0")
        eval_with_render(env, policy)
        # manually check
