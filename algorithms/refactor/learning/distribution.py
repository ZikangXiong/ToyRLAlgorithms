import torch
from torch.distributions import Categorical, normal


# ref: https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/6297608b8524774c847ad5cad87e14b80abf69ce/utilities/Utility_Functions.py#L22
def create_actor_distribution(action_type, actor_output, action_size):
    """Creates a distribution that the actor can then use to randomly draw actions"""
    if action_type == "discrete":
        assert actor_output.size()[1] == action_size, "Actor output the wrong size"
        action_distribution = Categorical(actor_output)  # this creates a distribution to sample from
    elif action_type == "continuous":
        assert actor_output.size()[1] == action_size * 2, "Actor output the wrong size"
        means = actor_output[:, :action_size].squeeze(0)
        stds = actor_output[:, action_size:].squeeze(0)
        # if len(means.shape) == 2:
        #     means = means.squeeze(-1)
        # if len(stds.shape) == 2:
        #     stds = stds.squeeze(-1)
        # if len(stds.shape) > 1 or len(means.shape) > 1:
        #     raise ValueError("Wrong mean and std shapes - {} -- {}".format(stds.shape, means.shape))
        action_distribution = normal.Normal(means.squeeze(0), torch.abs(stds).clamp(-0.01, 0.01))
    else:
        raise NotImplementedError()

    return action_distribution
