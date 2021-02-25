import torch as th


def change_optim_lr(optim: th.optim.Optimizer, lr):
    for param_group in optim.param_groups:
        param_group["lr"] = lr


def grad_clip(model, clip_val):
    for param in model.parameters():
        param.grad.data.clamp_(-clip_val, clip_val)


def polyak_update(original_model, target_model, tau):
    # polyak update
    for target_param, param in zip(target_model.parameters(), original_model.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
