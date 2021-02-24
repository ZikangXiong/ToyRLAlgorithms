import torch as th


def change_optim_lr(optim: th.optim.Optimizer, lr):
    for param_group in optim.param_groups:
        param_group["lr"] = lr
