"""How much of the block quantization error comes from the largest-magnitude element inflating the scaling factor?

Per block, the largest-magnitude element (argmax |x| inside the block -- NOT a statistical outlier) is kept in BF16
(its own error is only its BF16 rounding); the other elements are quantized to FP4 (E2M1) with the format's block scale. Two variants differ ONLY in which
elements define the scale:
  A "included": scale from the block amax INCLUDING the largest-magnitude element (standard scale; that element is just
                stored in BF16)
  B "excluded": scale from the block amax of the OTHER elements (the second-largest magnitude sets the scale)
plus the plain baseline (the largest-magnitude element quantized like everyone else). NVFP4's tensor-level scale s_enc is computed from the
ORIGINAL tensor and shared by all variants, so A and B differ in nothing but the block amax.

Relative MSE = per-block sum(err^2) / sum(x^2) over ALL elements of the block (largest-magnitude element included), averaged over blocks with
sum(x^2) > 0 -- the same definition as H2 (rel_mse). The denominators of A and B are identical, so B/A does not depend on it.

Modes:
  python3 scale_largest_magnitude.py laplace [millions_of_elements=64]     unit-variance i.i.d. Laplace
  python3 scale_largest_magnitude.py real <llama_base|qwen3_moe|deepseek_moe_16b>   real activations, exactly as H1 (X pooled over
                                                                          all Linear inputs, Q and K after RoPE, 2048 tokens)
  python3 scale_largest_magnitude.py check                                 correctness checks (no GPU needed)
"""
import csv
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from formats import CONFIGS, E4M3_MAX, _positive_fp_levels_f64, quantize_blocks_config, quantize_tensor_config  # noqa: E402

RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
FORMATS = {"nvfp4": (16, "E2M1_amax_E4M3"), "mxfp4": (32, "E2M1_mxfp4_ue8m0")}


def s_enc_of(blocks: torch.Tensor, cfg_name: str):
    """Tensor-level NVFP4 encode scale from the tensor's own amax (same formula as quantize_tensor_config)."""
    cfg = CONFIGS[cfg_name]
    if cfg["scale_selection"] == "amax" and cfg["scale_format"] == "e4m3":
        ref_levels, _ = _positive_fp_levels_f64(cfg.get("scale_ref_fmt") or cfg["fmt"], blocks.device)
        amax = torch.clamp(blocks.to(torch.float64).abs().amax(), min=2.0 ** -170)
        return (ref_levels[-1] * E4M3_MAX) / amax
    return None


def quant(blocks: torch.Tensor, cfg_name: str, s_enc) -> torch.Tensor:
    cfg = CONFIGS[cfg_name]
    return quantize_blocks_config(blocks, cfg["fmt"], cfg["scale_selection"], cfg["scale_format"], cfg.get("scale_ref_fmt"), s_enc)


@torch.no_grad()
def variants(blocks: torch.Tensor, cfg_name: str):
    """blocks [N, bs] float32 -> per-block relative MSE (plain, A, B) for the blocks with sum(x^2) > 0."""
    s_enc = s_enc_of(blocks, cfg_name)
    top2 = blocks.abs().topk(2, dim=-1)
    idx = top2.indices[:, :1]                                   # position of the largest-magnitude element
    largest = blocks.gather(1, idx)
    second = top2.values[:, 1:2]
    largest_bf16 = largest.to(torch.bfloat16).to(torch.float32)

    q_plain = quant(blocks, cfg_name, s_enc)
    q_a = q_plain.clone()
    q_a.scatter_(1, idx, largest_bf16)                          # A: scale includes it, element stored in BF16
    blocks_b = blocks.clone()
    blocks_b.scatter_(1, idx, torch.sign(largest) * second)     # it no longer sets the block amax
    q_b = quant(blocks_b, cfg_name, s_enc)
    q_b.scatter_(1, idx, largest_bf16)                          # B: scale from the others, element stored in BF16

    sig = blocks.pow(2).sum(-1)
    valid = sig > 0
    out = []
    for q in (q_plain, q_a, q_b):
        err = (blocks - q).pow(2).sum(-1)
        out.append((err[valid] / sig[valid]))
    return out


def stats(x: torch.Tensor, block_size: int, base_config: str):
    """Drop-in for H1's block_stats(x, block_size, base_config): returns SUMS keyed like H1's accumulator. The three H1
    fields are re-used as  match -> sum rel-MSE plain,  contribution -> sum rel-MSE A,  largest_err_contribution -> sum
    rel-MSE B,  n -> number of valid blocks (relabelled in the output CSV)."""
    K = x.shape[-1]
    n_blocks = K // block_size
    if n_blocks == 0:
        return None
    blocks = x.reshape(-1, K)[:, : n_blocks * block_size].to(torch.float32).reshape(-1, block_size)
    plain, a, b = variants(blocks, base_config)
    if plain.numel() == 0:
        return None
    return dict(match=plain.sum().item(), contribution=a.sum().item(), largest_err_contribution=b.sum().item(), n=int(plain.numel()))


def row_from(model, target, fmt, plain, a, b, n):
    return dict(model=model, target=target, format=fmt, relmse_plain=plain, relmse_A_scale_incl_largest=a,
                relmse_B_scale_excl_largest=b, reduction_A_vs_plain_pct=100 * (1 - a / plain),
                reduction_B_vs_A_pct=100 * (1 - b / a), n_blocks=n)


FIELDS = ["model", "target", "format", "relmse_plain", "relmse_A_scale_incl_largest", "relmse_B_scale_excl_largest",
          "reduction_A_vs_plain_pct", "reduction_B_vs_A_pct", "n_blocks"]


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved -> {path}", flush=True)


def show(row):
    print(f"[{row['model']}][{row['format']}] {row['target']:>3s}: rel MSE plain {row['relmse_plain']:.4f}  "
          f"A (scale incl. largest-magnitude) {row['relmse_A_scale_incl_largest']:.4f}  B (scale excl. largest-magnitude) "
          f"{row['relmse_B_scale_excl_largest']:.4f}   A vs plain {row['reduction_A_vs_plain_pct']:.2f}%  "
          f"B vs A {row['reduction_B_vs_A_pct']:.2f}%  (n={row['n_blocks']})", flush=True)


def run_laplace(n_total):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    b_scale = 1.0 / math.sqrt(2.0)  # unit variance
    rows = []
    for fmt, (bs, cfg) in FORMATS.items():
        gen = torch.Generator(device=device)
        gen.manual_seed(1111)
        sums, n, done = [0.0, 0.0, 0.0], 0, 0
        while done < n_total:
            m = min(8 * 2**20, n_total - done)
            u = torch.rand((m // bs, bs), generator=gen, device=device, dtype=torch.float64) - 0.5
            x = (-b_scale * torch.sign(u) * torch.log1p(-2.0 * u.abs())).to(torch.float32)
            for i, v in enumerate(variants(x, cfg)):
                sums[i] += v.sum().item()
            n += x.shape[0]
            done += m
        row = row_from("laplace(var=1)", "iid", fmt, sums[0] / n, sums[1] / n, sums[2] / n, n)
        rows.append(row)
        show(row)
    write_rows(os.path.join(RESULTS_DIR, "scale_largest_magnitude_laplace.csv"), rows)


class _DropH1Lines:
    """H1's measure_model prints its own metric names for the re-used fields; hide those lines."""
    def __init__(self, out):
        self.out = out
    def write(self, s):
        if "match_rate=" not in s:
            self.out.write(s)
    def flush(self):
        self.out.flush()


def run_real(model_key):
    import real
    real.block_stats = stats
    if model_key == "llama_base":
        measure = real.measure_model
    else:
        import real_moe
        real_moe.block_stats = stats
        measure = real_moe.measure_model
    old = sys.stdout
    sys.stdout = _DropH1Lines(old)
    try:
        res = measure(model_key) if model_key == "llama_base" else measure(model_key)
    finally:
        sys.stdout = old
    rows = []
    for (target, fmt), s in sorted(res.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        # H1 stores 100 * sum / n:  match_rate = 100*mean(plain), contribution = 100*mean(A), largest_error_contribution = 100*mean(B)
        rows.append(row_from(model_key, target, fmt, s["match_rate"] / 100, s["contribution"] / 100,
                             s["largest_err_contribution"] / 100, s["n"]))
        show(rows[-1])
    write_rows(os.path.join(RESULTS_DIR, f"scale_largest_magnitude_{model_key}.csv"), rows)


def run_checks():
    torch.manual_seed(0)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ok = True
    for fmt, (bs, cfg) in FORMATS.items():
        x = torch.distributions.Laplace(0.0, 1.0).sample((256, 8 * bs)).to(dev)
        blocks = x.reshape(-1, bs)
        # 1. the plain quantizer used here is bit-identical to the project's quantize_tensor_config
        ref = quantize_tensor_config(x, cfg, bs).reshape(-1, bs)
        mine = quant(blocks, cfg, s_enc_of(blocks, cfg))
        same = torch.equal(ref, mine)
        # 2. A differs from plain only at the largest-magnitude position (BF16-exact there)
        s_enc = s_enc_of(blocks, cfg)
        idx = blocks.abs().argmax(-1, keepdim=True)
        pl, a, b = variants(blocks, cfg)
        # 3. on average B <= A (a smaller scale can only refine the other elements), and a hand-made block behaves as expected
        toy = torch.tensor([[40.0] + [1.0, -0.7, 0.4, 0.2, -0.3, 0.9, 0.1, -1.0, 0.6, 0.5, -0.2, 0.8, 0.3, -0.6, 0.7] + [0.0] * (bs - 16)],
                           device=dev)
        tp, ta, tb = variants(toy, cfg)
        print(f"[check][{fmt}] plain identical to quantize_tensor_config: {same}; mean rel MSE plain {pl.mean():.4f} A {a.mean():.4f} "
              f"B {b.mean():.4f}; toy block (one large element, 40): plain {tp.item():.5f} A {ta.item():.5f} B {tb.item():.5f}")
        ok &= same and bool(b.mean() <= a.mean() <= pl.mean()) and bool(tb.item() < ta.item())
    print("CHECKS", "OK" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "check":
        sys.exit(0 if run_checks() else 1)
    elif mode == "laplace":
        run_laplace(int(float(sys.argv[2]) * 2**20) if len(sys.argv) > 2 else 64 * 2**20)
    elif mode == "real":
        run_real(sys.argv[2])
    else:
        raise SystemExit(__doc__)
