import argparse
import gym
import copy
import torch as th

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

HYPER_PARAM = {
    "dqn": {
        "Pong-v4": {
            "pixel_input": True,
            "buffer_size": 120000,
            "time_step": 100000,
            "n_step": 10000,
            "n_epoch": 3,
            "training_step": 10,
            "batch_size": 1512,
            "lr": 1e-4,
            "gamma": 0.99,
            "tau": 0.1,
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


def dqn_eval(env_name):
    from algorithms.dqn import DQN

    env = gym.make(env_name)
    dqn = DQN.load(f"{ROOT}/pretrain/{env_name}/dqn.pth", env=env)
    env = dqn.env

    for _ in range(5):
        obs = env.reset()
        env.render()
        ep_reward = 0
        while True:
            action = dqn.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            ep_reward += reward

            if done:
                print(f"reward: {ep_reward}")
                break


if __name__ == '__main__':
    dqn_train("Pong-v4")
    dqn_eval("Pong-v4")
