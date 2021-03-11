import torch as th


def log_actor_critic_graph(log_writer, env, policy, device):
    _input = env.observation_space.sample()
    _input = th.from_numpy(_input).float().to(device).unsqueeze(dim=0)
    log_writer.add_graph(policy, _input)
