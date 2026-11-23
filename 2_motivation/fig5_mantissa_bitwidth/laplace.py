"""Figure 5 (Laplace model): relative MSE reduction when the selected elements get mantissa bitwidth M1..M7
(E2M1..E2M7, exponent fixed at 2), on synthetic unit-variance Laplace data (dense.py / moe.py measure the
same on real activations).

In each block, the top 12.5% elements by base squared quantization error (k=2 of 16 for NVFP4, k=4 of 32
for MXFP4) take their E2M{m} value under the same block scale; the rest stay E2M1.
  R_base = relative MSE with all elements E2M1, R_m = relative MSE with the top-k elements at E2M{m},
  reduction(m) = 100 * (1 - R_m / R_base).

Samples ~ Laplace(0, 1/sqrt(2)) (unit variance), seed 1111, num_rows x 8 blocks.
NVFP4: E2M{m}_amax_E4M3 (block 16, two-level scale over the whole sample tensor).
MXFP4: E2M{m}_mxfp4_ue8m0 (block 32, block-local UE8M0 scale).

Usage: python3 laplace.py <nvfp4|mxfp4> [num_rows=2000000] [num_blocks=8]
Output: results/laplace_nvfp4.csv, results/laplace_mxfp4.csv
"""

import csv
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
from formats import quantize_tensor_config  # noqa: E402

UNIT_VARIANCE_LAPLACE_SCALE = 1.0 / (2.0 ** 0.5)
MANTISSA_VALUES = list(range(1, 8))
FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, k=2, config="E2M{m}_amax_E4M3", two_level=True),
    "mxfp4": dict(block_size=32, k=4, config="E2M{m}_mxfp4_ue8m0", two_level=False),
}
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


@torch.no_grad()
def relative_mse_topk(A: torch.Tensor, fmt: str, m: int) -> float:
    cfg = FORMAT_CONFIG[fmt]
    bs, k = cfg["block_size"], cfg["k"]
    A_base = quantize_tensor_config(A, cfg["config"].format(m=1), block_size=bs, two_level=cfg["two_level"])
    A_boost = quantize_tensor_config(A, cfg["config"].format(m=m), block_size=bs, two_level=cfg["two_level"])

    A_b = A.reshape(-1, bs)
    eps_base_sq = (A_b - A_base.reshape(-1, bs)).pow(2)
    eps_boost_sq = (A_b - A_boost.reshape(-1, bs)).pow(2)
    sig = A_b.pow(2).sum(dim=-1)

    order = eps_base_sq.argsort(dim=-1, descending=True)
    mask = torch.zeros_like(eps_base_sq, dtype=torch.bool)
    mask.scatter_(-1, order[:, :k], True)
    err = torch.where(mask, eps_boost_sq, eps_base_sq).sum(dim=-1)
    return (err / sig).mean().item()


def main():
    fmt = sys.argv[1]
    num_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
    num_blocks = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    k = FORMAT_CONFIG[fmt]["k"]

    torch.manual_seed(1111)
    laplace = torch.distributions.Laplace(0.0, UNIT_VARIANCE_LAPLACE_SCALE)
    A = laplace.sample((num_rows, num_blocks * FORMAT_CONFIG[fmt]["block_size"]))

    r_base = relative_mse_topk(A, fmt, 1)  # m=1: selected elements stay E2M1
    print(f"[{fmt}] R_base (m=1) = {100 * r_base:.4f}%", flush=True)

    rows = []
    for m in MANTISSA_VALUES:
        r = r_base if m == 1 else relative_mse_topk(A, fmt, m)
        red = 100 * (1 - r / r_base)
        print(f"[{fmt}] k={k} m={m}: R={100 * r:.4f}%  reduction_rate={red:.2f}%", flush=True)
        rows.append({"format": fmt, "k": k, "boost_mant_bits": m, "relative_mse_pct": 100 * r, "reduction_rate_pct": red})

    csv_path = os.path.join(RESULTS_DIR, f"laplace_{fmt}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["format", "k", "boost_mant_bits", "relative_mse_pct", "reduction_rate_pct"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
