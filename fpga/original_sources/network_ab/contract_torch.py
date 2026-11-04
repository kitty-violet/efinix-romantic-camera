"""Lazy PyTorch builder driven solely by model_b_contract.json.

Importing this module does not import/install torch. FP32 execution and autograd
still require the separately approved training environment; no QAT is claimed.
"""
from __future__ import annotations

if __package__:
    from .model_contract import QUANTIZED, compile_contract, load_contract
else:
    from model_contract import QUANTIZED, compile_contract, load_contract


def build_fp32_model(contract: dict | None = None):
    contract = load_contract() if contract is None else contract
    plan = compile_contract(contract)
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is unavailable; environment preparation is a separate authorized task") from exc

    class ContractModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.contract_sha256 = plan["contract_sha256"]
            self.contract_model_id = plan["model_id"]
            self.operators = nn.ModuleDict()
            self.names = {}
            for row in plan["nodes"]:
                if row["kind"] not in QUANTIZED:
                    continue
                safe_name = row["id"].replace(".", "__")
                self.names[row["id"]] = safe_name
                sequence = []
                if row["padding"] == "replicate_1":
                    sequence.append(nn.ReplicationPad2d(1))
                sequence.append(nn.Conv2d(row["input_channels"], row["channels"], row["kernel"],
                                          stride=row["stride"], padding=0, groups=row["groups"],
                                          bias=row["training_bias"]))
                bn = row["batch_norm"]
                if bn["enabled"]:
                    sequence.append(nn.BatchNorm2d(row["channels"], eps=bn["epsilon"],
                                                   momentum=bn["momentum"], affine=bn["affine"],
                                                   track_running_stats=bn["track_running_stats"]))
                if row["activation"] == "relu":
                    sequence.append(nn.ReLU(inplace=False))
                self.operators[safe_name] = nn.Sequential(*sequence)

        def forward(self, image, *, return_tensors=False):
            if image.ndim != 4 or image.shape[1] != 3 or image.shape[-2] % 4 or image.shape[-1] % 4:
                raise ValueError("NCHW RGB with H/W divisible by four required")
            values = {"input": image}
            for row in plan["nodes"]:
                source = values[row["inputs"][0]]
                if row["kind"] in QUANTIZED:
                    value = self.operators[self.names[row["id"]]](source)
                    if row["activation"] == "clamp_rgb":
                        value = torch.clamp(value, *contract["output"]["float_clamp"])
                elif row["kind"] == "nearest_x2":
                    # Explicit repetition avoids framework-dependent nearest
                    # coordinate options and is directly differentiable.
                    value = source.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
                else:
                    value = source + values[row["inputs"][1]]
                values[row["id"]] = value
            return values if return_tensors else values[contract["output"]["node"]]

        def deployment_eval(self):
            """Freeze running BN statistics; folding/QAT must be checked next."""
            return self.eval()

    model = ContractModel()
    if sum(parameter.numel() for parameter in model.parameters()) != plan["trainable_parameter_count"]:
        raise RuntimeError("PyTorch parameters disagree with the machine contract")
    return model
