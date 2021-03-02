import torch as th


def gae(t, states, rewards, value_net, lam, gamma):
    """
    When lambda equals to 0, gae equals to r_t + gamma V(s_{t+1}) - V(s_t)
    When lambda equals to 1, gae equals to MC estimation
    See page 5 of https://arxiv.org/pdf/1506.02438.pdf

    :param t:
    :param states:
    :param rewards:
    :param value_net:
    :param lam:
    :param gamma:
    :return:
    """
    assert len(rewards) - 2 - t >= 0

    _gae = 0
    with th.no_grad():
        values = value_net(states)
        for l in range(len(rewards) - 2 - t):
            _gae += (lam * gamma) ** l * delta(rewards, gamma, values, t, l)

    return _gae


def delta(rewards, gamma, values, t, l):
    assert len(rewards) == len(values)
    _delta = 0

    for i in range(l):
        _delta += gamma ** i * rewards[t + i]
    _delta += gamma ** l * rewards[t + l]

    return _delta
