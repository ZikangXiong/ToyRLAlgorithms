from typing import Tuple

import numpy as np


def single_worker_rollout(env, policy, buffer, rollout_steps):
    _steps = 0

    s1 = env.reset()
    while _steps < rollout_steps:
        a1 = policy.predict(s1)
        s2, r1, done, _ = env.step(a1)
        buffer.add(s1, a1, r1, done, s2)
        s1 = s2
        _steps += 1

        if done:
            s1 = env.reset()


def rollout_rew(buffer) -> Tuple:
    rews = np.array(buffer.buffer[2])
    dones = np.array(buffer.buffer[3])

    if len(rews) == 0:
        return None, None, None

    if dones[-1]:
        dones[-1] = False
    idx = np.where(dones)[0] + 1

    if len(idx) == 0:
        rew_mean = rews.sum()
        rew_min = rew_mean
        rew_max = rew_mean
    else:
        sub_arr_list = np.split(rews, idx)
        rews = [np.sum(item) for item in sub_arr_list]
        rew_mean = np.mean(rews)
        rew_min = np.min(rews)
        rew_max = np.max(rews)

    return rew_mean, rew_min, rew_max


def eval_with_render(env, policy, total_step=1000):
    policy.eval()
    rew_sum = 0

    obs = env.reset()
    env.render()
    for _ in range(total_step):
        action = policy.predict(obs)
        obs, rew, done, _ = env.step(action)
        rew_sum += rew
        env.render()

        if done:
            obs = env.reset()
            env.render()
    policy.train()

    return rew_sum
