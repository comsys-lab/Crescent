"""MXFP4 quantization (OCP Microscaling spec, as used for comparison against
NVFP4 -- see arXiv:2509.25149 Appendix B.4 for the exact numbers this
mirrors):

  - Element format: E2M1 (same 4-bit format as NVFP4: {0,0.5,1,1.5,2,3,4,6}).
  - Block size: 32 (NOT 16 -- NVFP4's block=16 is a deliberate narrowing
    from MXFP4's original 32, per the same paper).
  - Scale format: UE8M0 -- unsigned, pure power-of-two, no mantissa bits at
    all (unlike NVFP4's E4M3, which has both exponent and mantissa). Range
    2^-127 to 2^127.
  - Scale rounding: round UP (ceiling) to the next representable power of
    two, to guarantee the block's own max element never overflows (MXFP4
    has no tensor-level correction to fall back on if it rounded down).
  - No two-level (tensor + block) scaling -- MXFP4 is single-level, unlike
    NVFP4.

Reuses hypothesis_v2/common/formats.py's E2M1 level table and
quantize_to_fp_levels (hypothesis/common/quant.py) -- only the scale
computation (block=32, UE8M0 round-up) differs from NVFP4.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant import quantize_to_fp_levels, block_last_dimension
from formats import FMT_E2M1, _positive_fp_levels_f64

MXFP4_BLOCK_SIZE = 32
UE8M0_MIN_EXP = -127
UE8M0_MAX_EXP = 127

LEVELS, LSB_EVEN = _positive_fp_levels_f64(FMT_E2M1, torch.device("cpu"))
FP4_MAX = LEVELS[-1].item()  # 6.0


def round_up_ue8m0(scale: torch.Tensor) -> torch.Tensor:
    """UE8M0: pure power-of-two, no mantissa. Round UP (ceiling in log2
    space) to the next representable power of two >= scale, clamped to
    UE8M0's representable exponent range."""
    scale = torch.clamp(scale, min=2.0 ** (UE8M0_MIN_EXP))
    exp_up = torch.ceil(torch.log2(scale))
    exp_up = torch.clamp(exp_up, UE8M0_MIN_EXP, UE8M0_MAX_EXP)
    return torch.pow(2.0, exp_up)


def quantize_mxfp4_blocks(blocks: torch.Tensor) -> torch.Tensor:
    orig_dtype = blocks.dtype
    blocks64 = blocks.to(torch.float64)
    max_abs = blocks64.abs().amax(dim=-1, keepdim=True)
    ideal_scale = max_abs / FP4_MAX
    scale = round_up_ue8m0(ideal_scale)
    scale = torch.where(max_abs > 0, scale, torch.ones_like(scale))
    levels = LEVELS.to(blocks.device)
    lsb_even = LSB_EVEN.to(blocks.device)
    q = quantize_to_fp_levels(blocks64 / scale, levels, lsb_even) * scale
    return q.to(orig_dtype)


@torch.no_grad()
def quantize_tensor_mxfp4(tensor: torch.Tensor, block_size: int = MXFP4_BLOCK_SIZE) -> torch.Tensor:
    return block_last_dimension(tensor, block_size, quantize_mxfp4_blocks)
