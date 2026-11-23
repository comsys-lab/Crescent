"""Figure 2 (Laplace model): relative MSE reduction from relaxing each representation constraint of
NVFP4 and MXFP4 on synthetic unit-variance Laplace data (dense.py / moe.py measure the same on real
activations).

Each constraint is relaxed separately from the original format, the other two held as in the original:
  NVFP4 (block 16, E4M3 scale)                MXFP4 (block 32, E8M0 scale)
  R_BASE      = E2M1_amax_E4M3                R_BASE      = E2M1_mxfp4_ue8m0
  Recover_SF  = R_BASE - E2M1_optimal_none    Recover_SF  = R_BASE - E2M1_optimal_none
  Recover_Exp = R_BASE - E8M1_amax_E4M3       Recover_Exp = R_BASE - E8M1_mxfp4_ue8m0
  Recover_Man = R_BASE - E2M7_amax_E4M3       Recover_Man = R_BASE - E2M7_mxfp4_ue8m0
Relative MSE reduction rate (Figure 2) = Recover_* / R_BASE.

Samples ~ Laplace(0, 1/sqrt(2)) (unit variance), rounded to BF16 (the ground truth), seed 1111.
NVFP4 uses a two-level scale whose tensor-level part depends on the tensor amax, so the sample stream is
cut into chunks of TENSOR_SIZES elements (Linear tensor sizes of the 1.8M-67.9M range) and quantized
chunk by chunk; the three sizes give near-identical results. MXFP4's UE8M0 scale is block-local, so the
whole stream is quantized at once.

Usage: python3 laplace.py <nvfp4|mxfp4> [num_rows=2000000]
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
NUM_BLOCKS = 8  # blocks per sampled row
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16,
                  configs=["E2M1_amax_E4M3", "E2M1_optimal_none", "E8M1_amax_E4M3", "E2M7_amax_E4M3"]),
    "mxfp4": dict(block_size=32,
                  configs=["E2M1_mxfp4_ue8m0", "E2M1_optimal_none", "E8M1_mxfp4_ue8m0", "E2M7_mxfp4_ue8m0"]),
}
TENSOR_SIZES = {  # NVFP4 only
    "min_1.8M": 1_835_008,
    "mid_16.8M": 16_777_216,
    "max_67.9M": 67_895_296,
}


def sample(num_rows: int, block_size: int, seed: int = 1111) -> torch.Tensor:
    torch.manual_seed(seed)
    laplace = torch.distributions.Laplace(0.0, UNIT_VARIANCE_LAPLACE_SCALE)
    x = laplace.sample((num_rows, NUM_BLOCKS * block_size))
    return x.to(torch.bfloat16).to(torch.float32)


def mse_rel_nvfp4(config_name: str, num_rows: int, tensor_size: int) -> float:
    bs = FORMAT_CONFIG["nvfp4"]["block_size"]
    flat = sample(num_rows, bs).reshape(-1)
    chunk_elems = (tensor_size // bs) * bs  # whole blocks only
    err_sum, count = 0.0, 0
    for start in range(0, flat.numel(), chunk_elems):
        chunk = flat[start:start + chunk_elems]
        if chunk.numel() < bs:
            continue
        chunk = chunk[: (chunk.numel() // bs) * bs].reshape(-1, bs)
        chunk_q = quantize_tensor_config(chunk, config_name, block_size=bs)
        err = (chunk - chunk_q).pow(2).sum(dim=-1)
        sig = chunk.pow(2).sum(dim=-1)
        err_sum += (err / sig).sum().item()
        count += err.numel()
    return err_sum / count


def mse_rel_mxfp4(config_name: str, num_rows: int) -> float:
    bs = FORMAT_CONFIG["mxfp4"]["block_size"]
    blocks = sample(num_rows, bs).reshape(-1, bs)
    xq = quantize_tensor_config(blocks, config_name, block_size=bs, two_level=False)
    err = (blocks - xq).pow(2).sum(dim=-1)
    sig = blocks.pow(2).sum(dim=-1)
    return (err / sig).mean().item()


def decompose(fmt: str, values: dict) -> list:
    base, sf, exp, man = FORMAT_CONFIG[fmt]["configs"]
    r = values[base]
    rows = [
        {"component": f"R_{fmt.upper()}", "mse_rel_g": r},
        {"component": "Recover_SF", "mse_rel_g": r - values[sf]},
        {"component": "Recover_Exp", "mse_rel_g": r - values[exp]},
        {"component": "Recover_Man", "mse_rel_g": r - values[man]},
    ]
    for row in rows:
        print(f"[{fmt}] {row['component']:<12} = {100 * row['mse_rel_g']:.4f}%"
              f"  (reduction rate {100 * row['mse_rel_g'] / r:.2f}%)", flush=True)
    return rows


def main():
    fmt = sys.argv[1]
    num_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
    configs = FORMAT_CONFIG[fmt]["configs"]

    if fmt == "nvfp4":
        all_rows = []
        for label, tensor_size in TENSOR_SIZES.items():
            print(f"[nvfp4] tensor_size={tensor_size}", flush=True)
            values = {c: mse_rel_nvfp4(c, num_rows, tensor_size) for c in configs}
            for row in decompose(fmt, values):
                all_rows.append({"tensor_size_label": label, "tensor_size": tensor_size, **row})
        fieldnames = ["tensor_size_label", "tensor_size", "component", "mse_rel_g"]
    else:
        values = {c: mse_rel_mxfp4(c, num_rows) for c in configs}
        all_rows = decompose(fmt, values)
        fieldnames = ["component", "mse_rel_g"]

    csv_path = os.path.join(RESULTS_DIR, f"laplace_{fmt}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Saved -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
