"""CPU research blocks, not pretrained LLMs or a full Memory Mosaics v2 replica.

Uses the recurrent-key, future-value and memory-bandwidth equations from
https://arxiv.org/abs/2507.03285. No paper code or pretrained weights are copied.
"""
import math
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn


VARIANTS = ("attention", "mosaics", "mamba_attention", "mamba_mosaics",
            "rwkv7_attention", "rwkv7_mosaics")


def normalize(x):
    # Clamp before sqrt: a zero vector must not have a singular norm gradient.
    return x * mx.rsqrt(mx.maximum(mx.sum(x*x, axis=-1, keepdims=True), 1e-12))


class ContextMemory(nn.Module):
    """Symmetric causal retrieval with a gated recurrent key and delayed value."""

    def __init__(self, width, heads, window=None, delay=1):
        super().__init__()
        if (type(width) is not int or not 4 <= width <= 64 or type(heads) is not int
                or not 1 <= heads <= 8 or width % heads
                or (window is not None and (type(window) is not int or window < 2))
                or type(delay) is not int or delay < 1):
            raise ValueError("invalid bounded research memory dimensions or mask")
        self.width, self.heads = width, heads
        self.window, self.delay = window, delay
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.input_gate = nn.Linear(width, heads, bias=False)
        self.forget_gate = nn.Linear(width, heads, bias=False)
        self.beta0 = mx.full((heads,), 1.5)
        self.beta1 = mx.full((heads,), 1.5)
        self.alpha = mx.full((heads,), -math.log(2))
        self.gamma = mx.full((heads,), 0.5)
        self.value_scale = mx.zeros((heads,))

    def __call__(self, x):
        if (not isinstance(x, mx.array) or mx.default_device() != mx.cpu or x.ndim != 3 or x.shape[-1] != self.width
                or not 1 <= x.shape[0] <= 4 or not 1 <= x.shape[1] <= 64):
            raise ValueError("research memory requires CPU and [batch<=4, time<=64, width]")
        if x.dtype != mx.float32 or not bool(mx.all(mx.isfinite(x)).item()) or float(mx.max(mx.abs(x)).item()) > 64:
            raise ValueError("research memory input must be finite float32")
        b, t, _ = x.shape
        d = self.width // self.heads
        projected = self.key(x).reshape(b, t, self.heads, d)
        # Bounded exponent is a numerical safeguard, not the paper's initialization.
        gate = mx.exp(mx.clip(self.input_gate(x), -10, 10))
        forget = mx.exp(-mx.abs(self.forget_gate(x)))
        state = mx.zeros((b, self.heads, d), dtype=x.dtype)
        keys = []
        for index in range(t):
            state = gate[:, index, :, None] * projected[:, index] + forget[:, index, :, None] * state
            keys.append(normalize(state))
        keys = mx.stack(keys, axis=2)
        values = self.value(x).reshape(b, t, self.heads, d).transpose(0, 2, 1, 3)
        following = mx.concatenate([values[:, :, 1:], mx.zeros((b, self.heads, 1, d))], axis=2)
        gamma = self.gamma[None, :, None, None]
        values = gamma * values + (1 - gamma) * following
        values = normalize(values)
        values = values * mx.exp(mx.minimum(mx.abs(self.value_scale), 15))[None, :, None, None]
        positions = mx.arange(t)
        gap = positions[:, None] - positions[None, :]
        allowed = gap >= self.delay
        if self.window is not None:
            allowed = allowed & (gap < self.window)
        counts = mx.maximum(allowed.sum(axis=-1), 1)
        # shortcut: full O(T^2) memory, add scoped incremental caches before runtime use.
        beta = (mx.exp(mx.minimum(self.beta0, 10))[:, None]
                + mx.exp(mx.minimum(self.beta1, 10))[:, None]
                * counts[None, :] ** mx.sigmoid(self.alpha)[:, None])
        scores = (keys @ keys.transpose(0, 1, 3, 2)) * beta[None, :, :, None]
        weights = mx.softmax(mx.where(allowed[None, None], scores, -1e9), axis=-1)
        weights = weights * allowed[None, None]
        weights = weights / mx.maximum(weights.sum(axis=-1, keepdims=True), 1e-6)
        output = weights @ values
        return output.transpose(0, 2, 1, 3).reshape(b, t, self.width)


class HybridBlock(nn.Module):
    """Real MLX-LM recurrent blocks plus attention or a bounded Mosaics core."""

    def __init__(self, variant="mosaics", width=16, heads=2):
        super().__init__()
        if variant not in VARIANTS or mx.default_device() != mx.cpu:
            raise ValueError("choose a documented CPU research variant")
        if type(width) is not int or not 4 <= width <= 64 or type(heads) is not int or not 1 <= heads <= 8 or width % heads:
            raise ValueError("invalid research dimensions")
        self.variant, self.width = variant, width
        self.norm = nn.RMSNorm(width)
        if variant.startswith("mamba_"):
            from mlx_lm.models.mamba import MambaBlock, ModelArgs
            args = ModelArgs(model_type="mamba", vocab_size=8, hidden_size=width,
                             intermediate_size=2*width, state_size=4, num_hidden_layers=1,
                             conv_kernel=4, use_bias=False, use_conv_bias=True, time_step_rank=2)
            self.recurrent = MambaBlock(args)
        elif variant.startswith("rwkv7_"):
            # This owned CPU process uses the upstream recurrent fallback, not Metal.
            with patch.object(mx.metal, "is_available", return_value=False):
                from mlx_lm.models.rwkv7 import Rwkv7Layer, ModelArgs
            args = ModelArgs(model_type="rwkv7", vocab_size=8, hidden_size=width,
                             intermediate_size=2*width, norm_eps=1e-5, head_dim=width//heads,
                             num_hidden_layers=1, a_low_rank_dim=4, v_low_rank_dim=4,
                             gate_low_rank_dim=4, decay_low_rank_dim=4)
            self.recurrent = Rwkv7Layer(args, layer_idx=0)
            # This is a fresh toy initialization, not a pretrained RWKV checkpoint.
            self.recurrent.attn.k_k = mx.ones_like(self.recurrent.attn.k_k)
        if variant.endswith("mosaics"):
            self.short = ContextMemory(width, heads, window=8)
            self.long = ContextMemory(width, heads, delay=4)
            self.output = nn.Linear(2*width, width, bias=False)
        else:
            self.attention = nn.MultiHeadAttention(width, heads, bias=False)
        self.gate = nn.Linear(width, 2*width, bias=False)
        self.up = nn.Linear(width, 2*width, bias=False)
        self.down = nn.Linear(2*width, width, bias=False)

    def __call__(self, x):
        if (not isinstance(x, mx.array) or mx.default_device() != mx.cpu or x.ndim != 3 or x.shape[-1] != self.width
                or not 1 <= x.shape[0] <= 4 or not 1 <= x.shape[1] <= 64
                or x.dtype != mx.float32
                or not bool(mx.all(mx.isfinite(x)).item()) or float(mx.max(mx.abs(x)).item()) > 64):
            raise ValueError("bounded CPU probe requires finite [batch<=4, time<=64, width] input")
        z = self.norm(x)
        if self.variant.startswith("mamba_"):
            z = z + self.recurrent(z, cache=None)
        elif self.variant.startswith("rwkv7_"):
            z, _ = self.recurrent(z, v_first=None, cache=None)
        if self.variant.endswith("mosaics"):
            mixed = self.output(mx.concatenate([self.short(z), self.long(z)], axis=-1))
        else:
            mask = mx.where(mx.arange(x.shape[1])[:, None] >= mx.arange(x.shape[1])[None, :], 0, -1e9)
            mixed = self.attention(z, z, z, mask=mask)
        return x + mixed + self.down(nn.silu(self.gate(z)) * self.up(z))
