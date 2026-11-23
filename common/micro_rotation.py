"""Micro rotation: block-diagonal (group-wise) orthonormal Hadamard rotation along the last dimension.

"Micro" = the rotation group equals the quantization block (16 for NVFP4, 32 for MXFP4), as in MR-GPTQ
(Egiazarian/Panferov et al., arXiv:2509.23202, appendix K), instead of one Hadamard over the whole hidden
dimension (QuaRot-style full rotation). It works for every dimension divisible by the block, needs no
residual-stream / RMSNorm folding, and is applied online.

R = blockdiag(H_g, ..., H_g) with H_g the normalized Sylvester Hadamard (symmetric, H_g^T H_g = I), so
R^T = R = R^-1 and a matmul is unchanged when BOTH operands are rotated along the shared (contraction) dim:
    x @ W^T == (x R) @ (W R)^T          (Linear:  weight rotated offline, input rotated online)
    q . k   == (q R) . (k R)            (QK^T:    Q and K rotated by the same R, per head, after RoPE)

Everything is computed in float32; the result keeps the input dtype unless the caller wants float32.
"""
import math

import torch

_MATRICES = {}


def hadamard_matrix(g: int, device, dtype=torch.float32) -> torch.Tensor:
    if g <= 0 or (g & (g - 1)) != 0:
        raise ValueError(f"rotation group size must be a power of two, got {g}")
    key = (g, str(device), dtype)
    if key not in _MATRICES:
        h = torch.ones(1, 1, dtype=torch.float32)
        while h.shape[0] < g:
            h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
        _MATRICES[key] = (h / math.sqrt(g)).to(device=device, dtype=dtype)
    return _MATRICES[key]


def hadamard_transform_last_dim(x: torch.Tensor, g: int) -> torch.Tensor:
    """x -> x R over the last dim, group size g. Returns float32."""
    if g <= 1:
        return x.to(torch.float32)
    if x.shape[-1] % g != 0:
        raise ValueError(f"last dim {x.shape[-1]} is not divisible by rotation group size {g}")
    h = hadamard_matrix(g, x.device)
    y = x.to(torch.float32).reshape(*x.shape[:-1], x.shape[-1] // g, g) @ h
    return y.reshape(x.shape)


def rotate_hessian(h: torch.Tensor, g: int) -> torch.Tensor:
    """Hessian of the rotated inputs: sum (xR)^T (xR) = R^T H R (R symmetric), same math as rotating
    the calibration activations, without re-collecting them."""
    hr = hadamard_transform_last_dim(h, g)              # H R
    return hadamard_transform_last_dim(hr.t().contiguous(), g).t().contiguous()  # R^T (H R)


def with_rotation(fn, g: int):
    """fn is a quantizer x -> x_q. Returns x -> fn(x R) in x's dtype (rotation in fp32, quantizer sees fp32)."""
    def rotated(x: torch.Tensor) -> torch.Tensor:
        return fn(hadamard_transform_last_dim(x.detach(), g)).to(x.dtype)
    return rotated
