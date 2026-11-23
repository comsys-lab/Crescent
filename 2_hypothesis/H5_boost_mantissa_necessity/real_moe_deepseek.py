"""DeepSeek-MoE-16B-Base variant of real_moe_qwen3.py (same metric, same categories).
Model = common/deepseek_moe_model.py (pure-PyTorch port, verified identical to the original);
experts are plain nn.Linear so they are probed with ordinary forward-pre-hooks.

Usage: CUDA_VISIBLE_DEVICES=<free GPUs> python3 real_moe_deepseek.py <nvfp4|mxfp4>
"""
import csv
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_tokenizer_and_encodings
from formats import quantize_tensor_config, quantize_tensor_topk_activation_e2m7
import deepseek_moe_model as dsm
from deepseek_moe_model import MODEL_NAME

torch.manual_seed(1111)

MANTISSA_VALUES = [1, 2, 3, 4, 5, 6, 7]
PROBE_SEQ_LEN = 2048

FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, base_config="E2M1_amax_E4M3", k=2),
    "mxfp4": dict(block_size=32, base_config="E2M1_mxfp4_ue8m0", k=4),
}
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def iter_target_linears(model: nn.Module):
    """All nn.Linear probe points. DeepSeek-MoE experts are plain nn.Linear, and real.py's
    one-per-(layer, category) dedup would keep expert 0 only, so EVERY expert's gate_proj
    ("gateup_proj": shared input of the gate/up matmuls) and down_proj is hooked, like the
    Qwen3 script probes every active expert. q_proj stands for q/k/v ("qkv_proj": one shared
    input); Q/K themselves are probed after RoPE via dsm.set_hooks."""
    by_suffix = {"q_proj": "qkv_proj", "o_proj": "o_proj", "gate_proj": "gateup_proj", "down_proj": "down_proj"}
    targets = []
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear) or n == "lm_head":
            continue
        category = by_suffix.get(n.rsplit(".", 1)[-1])
        if category is not None:
            targets.append((n, m, category))
    return targets


def boost_config_name(fmt: str, mant_bits: int) -> str:
    return f"E2M{mant_bits}_amax_E4M3" if fmt == "nvfp4" else f"E2M{mant_bits}_mxfp4_ue8m0"


@torch.no_grad()
def reduction_pct_all_m(x: torch.Tensor, fmt: str, k: int, block_size: int, base_config: str) -> dict:
    x32 = x.detach().to(torch.float32)
    xb = quantize_tensor_config(x32, base_config, block_size)
    K = x32.shape[-1]
    n_blocks = K // block_size
    if n_blocks == 0:
        return None
    usable = n_blocks * block_size
    x_b = x32[..., :usable].reshape(-1, block_size)
    base_b = xb[..., :usable].reshape(-1, block_size)
    sig = x_b.pow(2).sum(dim=-1).clamp(min=1e-30)
    r_base = ((x_b - base_b).pow(2).sum(dim=-1) / sig).mean().item()
    if r_base <= 0:
        return None
    out = {}
    for m in MANTISSA_VALUES:
        xu = quantize_tensor_topk_activation_e2m7(x32, k, block_size=block_size,
                                                    base_config=base_config, boost_config=boost_config_name(fmt, m))
        xu_b = xu[..., :usable].reshape(-1, block_size)
        r_m = ((x_b - xu_b).pow(2).sum(dim=-1) / sig).mean().item()
        out[m] = 100 * (1 - r_m / r_base)
    return out


def identity(x):
    return x


def accumulate(cat_sum, cat_n, category, res):
    if res is None:
        return
    if category not in cat_sum:
        cat_sum[category] = {m: 0.0 for m in MANTISSA_VALUES}
        cat_n[category] = 0
    for m in MANTISSA_VALUES:
        cat_sum[category][m] += res[m]
    cat_n[category] += 1


@torch.no_grad()
def measure_model(fmt: str) -> dict:
    cfg = FORMAT_CONFIG[fmt]
    k, block_size, base_config = cfg["k"], cfg["block_size"], cfg["base_config"]

    same_seeds()
    full_text = load_wikitext2_text()
    _, encodings = load_tokenizer_and_encodings(MODEL_NAME, full_text)

    cat_sum, cat_n = {}, {}

    def q_probe(x):
        accumulate(cat_sum, cat_n, "Q", reduction_pct_all_m(x.detach(), fmt, k, block_size, base_config))
        return x

    def k_probe(x):
        accumulate(cat_sum, cat_n, "K", reduction_pct_all_m(x.detach(), fmt, k, block_size, base_config))
        return x

    def gateup_probe(x):
        accumulate(cat_sum, cat_n, "gateup_proj", reduction_pct_all_m(x.detach(), fmt, k, block_size, base_config))
        return x

    def down_probe(x):
        accumulate(cat_sum, cat_n, "down_proj", reduction_pct_all_m(x.detach(), fmt, k, block_size, base_config))
        return x

    dsm.set_hooks(q=q_probe, k=k_probe)  # Q/K probed after RoPE (see common/deepseek_moe_model.py)
    model = dsm.load_model()

    targets = iter_target_linears(model)
    print(f"Hooking {len(targets)} Linear modules (q/o_proj + every expert gate/down_proj)", flush=True)

    def make_hook(category):
        def hook(_mod, inputs):
            accumulate(cat_sum, cat_n, category, reduction_pct_all_m(inputs[0].detach(), fmt, k, block_size, base_config))
        return hook

    handles = [m.register_forward_pre_hook(make_hook(category)) for _n, m, category in targets]

    ids = encodings.input_ids[:, :PROBE_SEQ_LEN].to(model.device)
    model(input_ids=ids)
    dsm.set_hooks()
    for h in handles:
        h.remove()

    by_category = {}
    for category in sorted(cat_sum.keys()):
        avg = {m: cat_sum[category][m] / cat_n[category] for m in MANTISSA_VALUES}
        print(f"[{fmt}][deepseek_moe_16b] {category}: " + " ".join(f"m{m}={avg[m]:.2f}%" for m in MANTISSA_VALUES), flush=True)
        by_category[category] = avg

    del model
    cleanup_cuda()
    return by_category


def main():
    fmt = sys.argv[1]
    by_category = measure_model(fmt)

    csv_path = os.path.join(RESULTS_DIR, f"real_summary_{fmt}_deepseek_moe_16b.csv")
    categories = sorted(by_category.keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "m"] + categories)
        w.writeheader()
        for m in MANTISSA_VALUES:
            row = {"model": "deepseek_moe_16b", "m": m}
            for category in categories:
                row[category] = by_category[category].get(m, "")
            w.writerow(row)
    print(f"Saved -> {csv_path}", flush=True)
    print("H4_DEEPSEEK_MOE_DONE", flush=True)


if __name__ == "__main__":
    main()
