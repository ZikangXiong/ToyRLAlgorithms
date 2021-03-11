import torch as th


def change_optim_lr(optim, lr):
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


def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    if len(model_state_dict_1) != len(model_state_dict_2):
        print(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1:]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1:]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
            model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if th.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if th.cuda.is_available() else "cpu")

        if not th.allclose(v_1, v_2):
            print(f"Tensor mismatch: {v_1} vs {v_2}")
            return False

    return True
