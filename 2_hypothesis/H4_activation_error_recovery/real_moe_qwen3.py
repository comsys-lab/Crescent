"""H3 (v4), Qwen3-30B-A3B MoE variant -- Relative MSE Reduction Rate
(upgrading top-k largest-|eps_i^2| positions to E2M7, sweeping k=0..
block_size), same metric as real.py but for Q/K (attention, plain probe)
and gateup_proj/down_proj (MoE experts, probed inline -- see
common/qwen3_moe_targets.py's module docstring for why experts need a
dedicated patch instead of iter_target_linears/register_forward_pre_hook).
Q/K/V/O_proj ARE plain nn.Linear here and get the normal hook treatment.

Usage: CUDA_VISIBLE_DEVICES=<free GPUs only> python3 real_moe_qwen3.py <nvfp4|mxfp4>
"""
import csv
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_tokenizer_and_encodings
from formats import quantize_tensor_config
from qwen3_moe_targets import MODEL_NAME, patch_eager_qwen3_moe, patch_qwen3_moe_experts

torch.manual_seed(1111)

FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, base_config="E2M1_amax_E4M3", boost_config="E2M7_amax_E4M3"),
    "mxfp4": dict(block_size=32, base_config="E2M1_mxfp4_ue8m0", boost_config="E2M7_mxfp4_ue8m0"),
}
PROBE_SEQ_LEN = 2048
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def should_skip_linear(name: str) -> bool:
    return name == "lm_head" or name.endswith(".lm_head")


CATEGORY_SUFFIXES = [("q_proj", "qkv_proj"), ("k_proj", "qkv_proj"), ("v_proj", "qkv_proj"), ("o_proj", "o_proj")]


def categorize(name: str) -> str:
    for suffix, category in CATEGORY_SUFFIXES:
        if name.endswith(suffix):
            return category
    return "other"


def iter_target_linears(model: nn.Module):
    targets = []
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear) or should_skip_linear(n):
            continue
        category = categorize(n)
        if category == "other":
            continue
        targets.append((n, m, category))
    return targets


@torch.no_grad()
def reduction_rate_of(x: torch.Tensor, block_size: int, base_config: str, boost_config: str, k_values: list) -> dict:
    K = x.shape[-1]
    num_blocks = K // block_size
    if num_blocks == 0:
        return None
    x = x.reshape(-1, K)[:, : num_blocks * block_size].to(torch.float32)
    xq = quantize_tensor_config(x, base_config, block_size)
    xq_boost = quantize_tensor_config(x, boost_config, block_size)

    x_b = x.reshape(-1, block_size)
    eps = x_b - xq.reshape(-1, block_size)
    eps_boost = x_b - xq_boost.reshape(-1, block_size)
    eps_sq = eps.pow(2)
    sig = x_b.pow(2).sum(dim=-1).clamp(min=1e-30)

    r_base = (eps_sq.sum(dim=-1) / sig).mean().item()
    order = eps_sq.argsort(dim=-1, descending=True)

    result = {0: 0.0}
    for k in k_values:
        topk_idx = order[:, :k]
        mask = torch.zeros_like(eps_sq, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)
        r_k_num = torch.where(mask, eps_boost.pow(2), eps_sq).sum(dim=-1)
        r_k = (r_k_num / sig).mean().item()
        result[k] = 100 * (1 - r_k / r_base) if r_base > 0 else 0.0
    return result


def identity(x):
    return x


def accumulate(cat_sum, cat_n, category, res, k_values):
    if res is None:
        return
    if category not in cat_sum:
        cat_sum[category] = {k: 0.0 for k in [0] + k_values}
        cat_n[category] = 0
    for k in [0] + k_values:
        cat_sum[category][k] += res[k]
    cat_n[category] += 1


@torch.no_grad()
def measure_model(fmt: str) -> dict:
    cfg = FORMAT_CONFIG[fmt]
    block_size, base_config, boost_config = cfg["block_size"], cfg["base_config"], cfg["boost_config"]
    k_values = list(range(1, block_size + 1))

    same_seeds()
    full_text = load_wikitext2_text()
    _, encodings = load_tokenizer_and_encodings(MODEL_NAME, full_text)

    cat_sum, cat_n = {}, {}

    def q_probe(x):
        accumulate(cat_sum, cat_n, "Q", reduction_rate_of(x.detach(), block_size, base_config, boost_config, k_values), k_values)
        return x

    def k_probe(x):
        accumulate(cat_sum, cat_n, "K", reduction_rate_of(x.detach(), block_size, base_config, boost_config, k_values), k_values)
        return x

    def gateup_probe(x):
        accumulate(cat_sum, cat_n, "gateup_proj", reduction_rate_of(x.detach(), block_size, base_config, boost_config, k_values), k_values)
        return x

    def down_probe(x):
        accumulate(cat_sum, cat_n, "down_proj", reduction_rate_of(x.detach(), block_size, base_config, boost_config, k_values), k_values)
        return x

    import transformers.models.qwen3_moe.modeling_qwen3_moe as attn_mod
    attn_cls, orig_attn_forward = patch_eager_qwen3_moe(attn_mod, q_probe, k_probe, identity, identity)
    experts_cls, orig_experts_forward = patch_qwen3_moe_experts(attn_mod, gateup_probe, down_probe)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                                   low_cpu_mem_usage=True, attn_implementation="eager",
                                                   device_map="auto")
    model.eval()

    targets = iter_target_linears(model)
    print(f"Hooking {len(targets)} Linear modules (Q/K/V/O only)", flush=True)

    def make_hook(category):
        def hook(_mod, inputs):
            accumulate(cat_sum, cat_n, category, reduction_rate_of(inputs[0].detach(), block_size, base_config, boost_config, k_values), k_values)
        return hook

    handles = [m.register_forward_pre_hook(make_hook(category)) for _n, m, category in targets]

    ids = encodings.input_ids[:, :PROBE_SEQ_LEN].to(model.device)
    model(input_ids=ids)
    attn_cls.forward = orig_attn_forward
    experts_cls.forward = orig_experts_forward
    for h in handles:
        h.remove()

    by_category = {}
    for category in sorted(cat_sum.keys()):
        avg = {k: cat_sum[category][k] / cat_n[category] for k in [0] + k_values}
        print(f"[{fmt}][qwen3_moe] {category} ({cat_n[category]}): " + " ".join(f"k{k}={avg[k]:.2f}%" for k in k_values), flush=True)
        by_category[category] = avg

    del model
    cleanup_cuda()
    return by_category


def main():
    fmt = sys.argv[1]
    k_values = list(range(1, FORMAT_CONFIG[fmt]["block_size"] + 1))
    by_category = measure_model(fmt)

    csv_path = os.path.join(RESULTS_DIR, f"real_{fmt}_qwen3_moe.csv")
    categories = sorted(by_category.keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "k"] + categories)
        w.writeheader()
        for k in [0] + k_values:
            row = {"model": "qwen3_moe", "k": k}
            for category in categories:
                row[category] = by_category[category].get(k, "")
            w.writerow(row)
    print(f"Saved -> {csv_path}", flush=True)
    print("H3_QWEN3_MOE_DONE", flush=True)


if __name__ == "__main__":
    main()
