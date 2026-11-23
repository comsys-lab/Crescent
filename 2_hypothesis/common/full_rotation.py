"""Full-dimension Hadamard rotation (QuaRot-style) along the last dimension, for any width that has a
Hadamard structure, including widths that are not powers of two.

R is orthogonal (R R^T = I) and x -> x R keeps every matmul intact when the other operand is rotated the same way:
    x @ W^T == (x R) @ (W R)^T           q . k == (q R) . (k R)
R = F_1 (x) F_2 (x) ... (x) F_k (Kronecker product) with small dense orthonormal factors:
  * powers of two: Sylvester Hadamard, split into factors of at most 128;
  * n = 2^a * m with m odd or otherwise without a Hadamard matrix (Llama 14336 = 2^11*7, Qwen 768 = 2^8*3,
    DeepSeek 1408 = 2^7*11, 2816 = 2^8*11, 10944 = 2^6*171): the smallest m' = m*2^j that is a Hadamard order
    built by the Paley construction (28, 12, 44, 44, 684) times a Sylvester matrix of the remaining size.
Paley I needs a prime q = m'-1 with q = 3 (mod 4); Paley II a prime q = m'/2-1 with q = 1 (mod 4). Every matrix is
checked for orthogonality when it is built. Not necessarily symmetric, so the inverse is x -> x R^T.

Everything runs in float32 and returns float32.
"""
import math

import torch

_FACTORS = {}   # n -> list of float64 CPU factors
_ON_DEVICE = {}  # (n, device) -> list of float32 factors


def _is_prime(q: int) -> bool:
    if q < 2:
        return False
    i = 2
    while i * i <= q:
        if q % i == 0:
            return False
        i += 1
    return True


def _jacobsthal(q: int) -> torch.Tensor:
    residues = {(i * i) % q for i in range(1, q)}
    chi = torch.zeros(q, dtype=torch.float64)
    for a in range(1, q):
        chi[a] = 1.0 if a in residues else -1.0
    idx = (torch.arange(q).view(-1, 1) - torch.arange(q).view(1, -1)) % q
    return chi[idx]


def paley_hadamard(m: int):
    """+-1 Hadamard matrix of order m by Paley I / II, or None if this construction does not give order m."""
    if m > 1 and m % 4 != 0:
        return None
    q = m - 1
    if _is_prime(q) and q % 4 == 3:  # Paley I
        s = torch.zeros(m, m, dtype=torch.float64)
        s[0, 1:] = 1.0
        s[1:, 0] = -1.0
        s[1:, 1:] = _jacobsthal(q)
        h = torch.eye(m, dtype=torch.float64) + s
    elif m % 2 == 0 and _is_prime(m // 2 - 1) and (m // 2 - 1) % 4 == 1:  # Paley II
        q = m // 2 - 1
        s = torch.zeros(q + 1, q + 1, dtype=torch.float64)
        s[0, 1:] = 1.0
        s[1:, 0] = 1.0
        s[1:, 1:] = _jacobsthal(q)
        a = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
        b = torch.tensor([[1.0, -1.0], [-1.0, -1.0]], dtype=torch.float64)
        h = torch.kron(s, a) + torch.kron(torch.eye(q + 1, dtype=torch.float64), b)
    else:
        return None
    if not torch.allclose(h @ h.t(), m * torch.eye(m, dtype=torch.float64), atol=1e-9):
        return None
    return h


def _sylvester(p: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < p:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h


def _split_power_of_two(p: int, limit: int = 128):
    out = []
    while p > limit:
        out.append(limit)
        p //= limit
    if p > 1:
        out.append(p)
    return out


def factors(n: int):
    """Dense orthonormal factors whose Kronecker product is R for width n."""
    if n in _FACTORS:
        return _FACTORS[n]
    p = n & (-n)          # largest power of two dividing n
    m = n // p
    if m == 1:
        fs = [_sylvester(s) / math.sqrt(s) for s in _split_power_of_two(p)]
    else:
        fs, j = None, 0
        while (m << j) <= n and p % (1 << j) == 0:
            h = paley_hadamard(m << j)
            if h is not None:
                mm = m << j
                fs = [h / math.sqrt(mm)] + [_sylvester(s) / math.sqrt(s) for s in _split_power_of_two(p >> j)]
                break
            j += 1
        if fs is None:
            raise ValueError(f"no Hadamard structure found for width {n} (= {p} * {m})")
    total = 1
    for f in fs:
        total *= f.shape[0]
    assert total == n, (n, [f.shape[0] for f in fs])
    _FACTORS[n] = fs
    return fs


def _dev_factors(n: int, device):
    key = (n, str(device))
    if key not in _ON_DEVICE:
        _ON_DEVICE[key] = [f.to(device=device, dtype=torch.float32) for f in factors(n)]
    return _ON_DEVICE[key]


def _apply(x: torch.Tensor, transpose: bool) -> torch.Tensor:
    n = x.shape[-1]
    fs = _dev_factors(n, x.device)
    dims = [f.shape[0] for f in fs]
    y = x.to(torch.float32).reshape(*x.shape[:-1], *dims)
    nb = len(x.shape) - 1
    for i, f in enumerate(fs):
        f = f.t() if transpose else f
        y = torch.matmul(y.movedim(nb + i, -1), f).movedim(-1, nb + i)   # along axis i: row vector times F_i
    return y.reshape(x.shape)


def full_rotate_last_dim(x: torch.Tensor) -> torch.Tensor:
    """x -> x R over the whole last dimension."""
    return _apply(x, False)


def full_unrotate_last_dim(x: torch.Tensor) -> torch.Tensor:
    """x -> x R^T (the inverse)."""
    return _apply(x, True)


def full_rotate_hessian(h: torch.Tensor) -> torch.Tensor:
    """R^T H R = Hessian of the rotated inputs (H symmetric)."""
    hr = full_rotate_last_dim(h)                                   # H R
    return full_rotate_last_dim(hr.t().contiguous()).t().contiguous()   # (R^T H^T R)^T


def with_full_rotation(fn):
    """fn is a quantizer x -> x_q. Returns x -> fn(x R) in x's dtype (fp32 rotation, fp32 into the quantizer)."""
    def rotated(x: torch.Tensor) -> torch.Tensor:
        return fn(full_rotate_last_dim(x.detach())).to(x.dtype)
    return rotated
