"""Figure 3 (Laplace model): cumulative squared quantization error contribution of the top-k largest-error
elements in each NVFP4 / MXFP4 block, on synthetic unit-variance Laplace data (dense.py / moe.py measure
the same on real activations).

  ratio(k) = E_blocks[ sum_{i in S_k} eps_i^2 / sum_i eps_i^2 ],  eps_i = a_i - Q(a)_i,
  S_k = the k elements with the largest eps_i^2 in the block, k = 1..block size.

Samples ~ Laplace(0, 1/sqrt(2)) (unit variance), seed 1111, num_rows x 8 blocks.
NVFP4: E2M1_amax_E4M3 (block 16, two-level scale over the whole sample tensor).
MXFP4: E2M1_mxfp4_ue8m0 (block 32, block-local UE8M0 scale).

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
FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, config="E2M1_amax_E4M3", two_level=True),
    "mxfp4": dict(block_size=32, config="E2M1_mxfp4_ue8m0", two_level=False),
}
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def simulate(fmt: str, num_rows: int, num_blocks: int) -> list:
    cfg = FORMAT_CONFIG[fmt]
    bs = cfg["block_size"]
    torch.manual_seed(1111)
    laplace = torch.distributions.Laplace(0.0, UNIT_VARIANCE_LAPLACE_SCALE)
    A = laplace.sample((num_rows, num_blocks * bs))
    Aq = quantize_tensor_config(A, cfg["config"], block_size=bs, two_level=cfg["two_level"])

    eps_sq = (A - Aq).reshape(-1, bs).pow(2)
    eps_sq_sorted, _ = eps_sq.sort(dim=-1, descending=True)
    cum = eps_sq_sorted.cumsum(dim=-1)
    total = cum[..., -1:].clamp(min=1e-30)
    ratio_topk = (100 * cum / total).mean(dim=0)

    rows = []
    for k in range(1, bs + 1):
        pct = ratio_topk[k - 1].item()
        equal_dist_pct = 100 * k / bs
        print(f"[{fmt}] top-{k:2d} error ratio = {pct:6.2f}%  equal distribution = {equal_dist_pct:6.2f}%", flush=True)
        rows.append({"k": k, "top_k_error_ratio_pct": pct, "equal_distribution_pct": equal_dist_pct})
    return rows


def main():
    fmt = sys.argv[1]
    num_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
    num_blocks = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    rows = simulate(fmt, num_rows, num_blocks)
    csv_path = os.path.join(RESULTS_DIR, f"laplace_{fmt}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "top_k_error_ratio_pct", "equal_distribution_pct"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
