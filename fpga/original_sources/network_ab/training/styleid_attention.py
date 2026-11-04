"""Memory-bounded StyleID self-attention injection for local SD1.5.

Adapted from jiwoogit/StyleID (MIT), revision
d4ea655dc1f7b19e0eac9553ad087132d18bb96f, diffusers_implementation.
The upstream license and unmodified sources accompany the experiment.
Q/K/V live in CPU RAM, never a multi-GB per-image disk cache. PyTorch SDPA
evaluates the same temperature-scaled attention without the full N x N matrix.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


class InjectionState:
    def __init__(self):
        self.mode = "off"
        self.timestep = 0
        self.gamma = 0.75
        self.temperature = 1.5
        self.content = {}
        self.style = {}

    def cache_bytes(self):
        return sum(t.numel() * t.element_size()
                   for bank in (self.content, self.style)
                   for tensors in bank.values() for t in tensors)


class StyleIDProcessor:
    def __init__(self, state, name):
        self.state, self.name = state, name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        # Only installed on SD1.5 self-attention. Reject untested architectures.
        if encoder_hidden_states is not None or hidden_states.ndim != 3:
            raise ValueError("StyleID processor expects SD1.5 3D self-attention")
        if attn.spatial_norm is not None or attn.group_norm is not None:
            raise ValueError("Unexpected attention normalization")
        residual = hidden_states
        b, n, _ = hidden_states.shape
        q, k, v = (projection(hidden_states)
                   for projection in (attn.to_q, attn.to_k, attn.to_v))
        d = q.shape[-1] // attn.heads
        q, k, v = (t.view(b, n, attn.heads, d).transpose(1, 2) for t in (q, k, v))
        slot = (self.name, self.state.timestep)
        if self.state.mode == "content":
            self.state.content[slot] = (q.detach().to("cpu", copy=True),)
        elif self.state.mode == "style":
            self.state.style[slot] = tuple(t.detach().to("cpu", copy=True) for t in (k, v))
        elif self.state.mode == "inject":
            qc = self.state.content[slot][0].to(q.device)
            ks, vs = (t.to(q.device) for t in self.state.style[slot])
            q = (self.state.gamma * qc + (1 - self.state.gamma) * q)
            q = q * self.state.temperature
            k, v = ks, vs
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, k.shape[-2], b)
            attention_mask = attention_mask.view(b, attn.heads, -1, k.shape[-2])
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                                             dropout_p=0.0, scale=attn.scale)
        out = out.transpose(1, 2).reshape(b, n, attn.heads * d).to(q.dtype)
        out = attn.to_out[1](attn.to_out[0](out))
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


def install_processors(unet, state, layers=(7, 8, 9, 10, 11)):
    names = []
    for i in layers:
        attention = unet.up_blocks[i // 3].attentions[i % 3].transformer_blocks[0].attn1
        name = f"layer{i}_attn"
        attention.set_processor(StyleIDProcessor(state, name))
        names.append(name)
    return names
