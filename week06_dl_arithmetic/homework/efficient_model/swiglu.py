"""
gpt-oss style SwiGLU Feed-Forward Network

Reference SwiGLU implementation:
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings, ensure_contiguous


@triton.jit
def silu(x, alpha):
    # TODO: Replace with x * sigmoid(x * alpha)
    return x * tl.sigmoid(x * alpha)


@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr, alpha: tl.constexpr, limit: tl.constexpr):
    # a = gate, b = up
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    # TODO: Add clamping
    # TODO: Replace silu(a_row) * b_row with glu * (b_row + 1)
    c_row = silu(a_row, alpha).cast(b_row.dtype) * (b_row + 1)
    c_row = tl.maximum(tl.minimum(c_row, limit), -limit)
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


@triton.jit
def _swiglu_backward_kernel(dc_ptr, a_ptr, b_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr, alpha: tl.constexpr, limit: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)

    # recomputation to save memory
    # TODO: Update backward pass for gpt-oss style implementation (formula will be different!)
    sig_a = tl.sigmoid(a_row * alpha)
    glu_a = a_row * sig_a
    y = glu_a * (b_row + 1.0)

    clamp_mask = (y >= -limit) & (y <= limit)
    dc_clamped = tl.where(clamp_mask, dc_row, 0.0)

    db_row = dc_clamped * glu_a
    da_row = dc_clamped * (glu_a * (1.0 - sig_a) * alpha + sig_a) * (b_row + 1.0)

    tl.store(a_ptr + col_offsets, da_row, mask=mask)
    tl.store(b_ptr + col_offsets, db_row, mask=mask)
    tl.store(dc_ptr + col_offsets, y, mask=mask)
    # HINT: We're already recomputing values here.
    # Could a third store here help avoid saving something else?


def swiglu_forward(a, b, alpha, limit):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        alpha=alpha,
        limit=limit
    )
    return a, b, c.view(*ori_shape)


def swiglu_backward(a, b, dc, alpha, limit):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_backward_kernel[(n_rows,)](
        dc,
        a,
        b,
        dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        alpha=alpha,
        limit=limit
    )
    return a.view(*ori_shape), b.view(*ori_shape), dc.view(*ori_shape)


class MemoryEfficientSwiGLUMLP(torch.autograd.Function):
    """
    Memory-optimized SwiGLU MLP with selective recomputation.
    """
    
    @staticmethod
    def forward(ctx, x, w_gate, w_up, w_down, alpha, limit):
        gate = x @ w_gate.T
        up = x @ w_up.T

        # TODO: Replace with fused swiglu_forward kernel
        gate, up, activation_out = swiglu_forward(gate, up, alpha, limit)

        out = activation_out @ w_down.T

        # TODO: Save tensors for backward
        ctx.save_for_backward(x, gate, up, w_gate, w_up, w_down)  # TODO: fill this
        ctx.alpha = alpha
        ctx.limit = limit

        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Implement backward pass
        # ... unpack, recompute whats needed, d_activation ...
        x, gate, up, w_gate, w_up, w_down = ctx.saved_tensors
        alpha = ctx.alpha
        limit = ctx.limit

        orig_shape = grad_output.shape
        grad_output = grad_output.reshape(-1, orig_shape[-1])
        x  = x.reshape(-1, x.shape[-1])
        # swiglu_activation_out = swiglu_activation_out.reshape(-1, swiglu_activation_out.shape[-1])

        # Gradient through activation
        # ... = swiglu_backward(...)
        grad_activation_down = grad_output @ w_down

        grad_activation_gate_out, grad_activation_up_out, swiglu_activation_out = swiglu_backward(gate, up, grad_activation_down, alpha, limit)
        swiglu_activation_out = swiglu_activation_out.reshape(-1, swiglu_activation_out.shape[-1])

        grad_w_down = grad_output.T @ swiglu_activation_out

        del grad_activation_down
        del swiglu_activation_out
        
        grad_w_gate = grad_activation_gate_out.T @ x
        # grad_activation_gate_in = grad_activation_gate_out @ w_gate

        grad_w_up = grad_activation_up_out.T @ x
        # grad_activation_up_in = grad_activation_up_out @ w_up
        
        grad_x = grad_activation_gate_out @ w_gate
        grad_x = grad_x + grad_activation_up_out @ w_up 

        grad_alpha = None
        grad_limit = None

        grad_x = grad_x.reshape(orig_shape)
        # Gradients for w_down, w_gate, w_up, dx
        return grad_x, grad_w_gate, grad_w_up, grad_w_down, grad_alpha, grad_limit


class SwiGLUFeedForward(nn.Module):
    """
    gpt-oss style SwiGLU.
    
    output = W_down @ ((up + 1) * gate * sigmoid(gate * alpha))
    """
    
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.alpha = 1.702
        self.limit = 7.0

        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MemoryEfficientSwiGLUMLP.apply(
            x, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight, self.alpha, self.limit
        )
