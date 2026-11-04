"""Training-only parameter initialization and BN policy; the contract owns all ops."""
import torch
from torch import nn


def initialize_rgb_path(model):
    """Seed the existing six signed RGB lanes, with zero residual endpoints.

    Under identity running statistics the initial output is stride-4 RGB sampling
    followed by exact nearest repeat. This is an initializer, NOT trained output.
    All existing weights and BN affine parameters remain trainable.
    """
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.zero_()
                module.running_var.fill_(1-module.eps)
                module.num_batches_tracked.zero_()
                module.weight.fill_(1)
                module.bias.zero_()
        for name, sequence in model.operators.items():
            conv = next(m for m in sequence if isinstance(m, nn.Conv2d))
            if name.startswith("res"):
                if name.endswith("conv2__pw"):
                    next(m for m in sequence if isinstance(m, nn.BatchNorm2d)).weight.zero_()
                continue
            conv.weight.zero_()
            if conv.bias is not None:
                conv.bias.zero_()
            if name == "stem":
                for out in range(conv.out_channels):
                    lane = out % 6
                    conv.weight[out, lane % 3, 1, 1] = 1 if lane < 3 else -1
            elif name == "rgb_head":
                for channel in range(3):
                    conv.weight[channel, channel, 0, 0] = 1
                    conv.weight[channel, channel+3, 0, 0] = -1
            elif conv.groups == conv.in_channels:
                conv.weight[:, 0, 1, 1] = 1
            else:
                for out in range(conv.out_channels):
                    conv.weight[out, out % 6, 0, 0] = 1


def initialize_model(model, config):
    choice = config.get("initialization", "pytorch_default")
    if choice == "rgb_path_v1":
        if config.get("bn_policy") != "frozen_running_stats_train_affine":
            raise ValueError("RGB initializer requires its validated frozen-stat policy")
        initialize_rgb_path(model)
    elif choice != "pytorch_default":
        raise ValueError(f"Unknown initialization: {choice}")
    set_optimization_mode(model, config)


def set_optimization_mode(model, config):
    model.train()
    policy = config.get("bn_policy", "batch_running_stats")
    if policy == "frozen_running_stats_train_affine":
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()  # eval freezes statistics, not affine parameter autograd.
    elif policy != "batch_running_stats":
        raise ValueError(f"Unknown BN policy: {policy}")
