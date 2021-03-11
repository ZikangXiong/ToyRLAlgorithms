import gym


def infer_action_type(env):
    if type(env.action_space) is gym.spaces.Box:
        actor_type = "continuous"
        action_size = env.action_space.shape[-1]
    else:
        raise NotImplementedError("Only support continuous policy for now")

    return actor_type


def infer_action_size(env, action_type):
    if action_type == "continuous":
        action_size = env.action_space.shape[-1]
    else:
        raise NotImplementedError("Only support continuous policy for now")

    return action_size

