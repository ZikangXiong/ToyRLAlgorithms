import argparse
import gym
import copy

import os
ROOT = os.path.dirname(os.path.abspath(__file__))

HYPER_PARAM = {
    "dqn": {
        "Pong-v4": {
            "pixel_input": True,
            "buffer_size": 1000000,
            "time_step": 100000,
            "n_step": 2000,
            "n_epoch": 3,
            "training_step": 2,
            "batch_size": 64,
            "lr": 1e-3,
            "gamma": 0.99,
            "tau": 0.9,
            "grad_clip": 1.0
        }
    }
}


def dqn_train(env_name, device="cpu", seed=0):
    from algorithms.dqn import DQN

    hyper = copy.deepcopy(HYPER_PARAM["dqn"][env_name])
    pixel_input = hyper.pop("pixel_input")
    buffer_size = hyper.pop("buffer_size")

    env = gym.make(env_name)

    dqn = DQN(env, pixel_input, buffer_size, device, seed=seed)
    dqn.learn(**hyper)

    dqn.save(f"{ROOT}/pretrain/{env_name}/dqn.pth")


if __name__ == '__main__':
    dqn_train("Pong-v4")