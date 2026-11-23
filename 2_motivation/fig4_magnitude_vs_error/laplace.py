"""Figure 4 (Laplace model) on synthetic i.i.d. Laplace data (dense.py / moe.py measure the same on real activations).

Same three metrics, computed by the SAME block_stats() as the real-activation runs:
  match_rate                   : P(argmax_i |a_i| == argmax_i eps_i^2) over blocks (eps = quantization error)
  blockmax_error_contribution  : mean of eps_{argmax|a|}^2 / sum_i eps_i^2 over blocks
  largest_error_contribution   : mean of max_i eps_i^2 / sum_i eps_i^2 over blocks (the ceiling)
for NVFP4 (block 16, E4M3 scale) and MXFP4 (block 32, UE8M0 scale). Elements ~ UNIT-VARIANCE Laplace: density
exp(-|x|/b)/(2b) with b = 1/sqrt(2) (variance 2 b^2 = 1). Written to results/laplace_summary.csv (NOT real_*.csv, so merge_results.py leaves it alone).

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 laplace.py [n_elements_millions=64]
"""
import csv
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense import block_stats, FORMAT_CONFIG, RESULTS_DIR, DEVICE  # noqa: E402

CHUNK = 8 * 2**20  # elements per chunk (keeps GPU memory small)
SCALE_B = 1.0 / math.sqrt(2.0)  # Laplace scale b for unit variance: Var = 2 b^2 = 1


def laplace(shape, gen, device):
    u = torch.rand(shape, generator=gen, device=device, dtype=torch.float64) - 0.5      # U(-0.5, 0.5)
    x = -SCALE_B * torch.sign(u) * torch.log1p(-2.0 * u.abs())                          # inverse CDF
    return x.to(torch.float32)


def main():
    n_total = int(float(sys.argv[1]) * 2**20) if len(sys.argv) > 1 else 64 * 2**20
    rows = []
    for fmt, cfg in FORMAT_CONFIG.items():
        bs = cfg["block_size"]
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(1111)
        acc = dict(match=0.0, contribution=0.0, largest_err_contribution=0.0, n=0)
        done = 0
        while done < n_total:
            m = min(CHUNK, n_total - done)
            x = laplace((m // bs, bs), gen, DEVICE)          # one row per block
            if done == 0:
                print(f"[{fmt}] sample variance of the first chunk: {x.double().var().item():.4f} (target 1)", flush=True)
            s = block_stats(x, bs, cfg["base_config"])
            for k in ("match", "contribution", "largest_err_contribution", "n"):
                acc[k] += s[k]
            done += m
        n = acc["n"]
        row = dict(model="laplace(var=1)", target="iid", format=fmt, match_rate=100 * acc["match"] / n,
                   blockmax_error_contribution=100 * acc["contribution"] / n,
                   largest_error_contribution=100 * acc["largest_err_contribution"] / n, n_blocks=n)
        rows.append(row)
        print(f"[{fmt}] unit-variance Laplace i.i.d.: match_rate={row['match_rate']:.2f}%  blockmax_error_contribution="
              f"{row['blockmax_error_contribution']:.2f}%  largest_error_contribution={row['largest_error_contribution']:.2f}%  "
              f"(n={n} blocks)", flush=True)
    path = os.path.join(RESULTS_DIR, "laplace_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "target", "format", "match_rate", "blockmax_error_contribution",
                                          "largest_error_contribution", "n_blocks"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
