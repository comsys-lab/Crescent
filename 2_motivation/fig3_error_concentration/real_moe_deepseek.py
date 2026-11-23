"""H3 (top-k error ratio) for DeepSeek-MoE-16B-Base: what fraction of a block's
total quantization error is captured by its top-k largest-error positions?
Same methodology and categories as real.py / real_moe_qwen3.py.

Uses common/deepseek_moe_model.py (pure-PyTorch port; the checkpoint's remote
code does not import on transformers 5.17). Every routed expert, the shared
experts and the dense layer-0 MLP are plain nn.Linear, so the probes are plain
forward-pre-hooks. real.py's iter_target_linears keeps only ONE Linear per
(layer, category), which here would be expert 0 alone (it sees a small,
unrepresentative slice of the tokens); like real_moe_qwen3.py, which probes
every active expert on every call, this file hooks EVERY expert's gate_proj
(the shared input of the fused gate/up matmul -> "gateup_proj") and down_proj
("down_proj"). Attention: q_proj is hooked once as "qkv_proj" (q/k/v share one
input), o_proj as "o_proj"; Q and K are probed after RoPE via set_hooks.

Usage: CUDA_VISIBLE_DEVICES=<free GPUs> python3 real_moe_deepseek.py <nvfp4|mxfp4>
"""
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_tokenizer_and_encodings
from formats import quantize_tensor_config
import deepseek_moe_model as dsm
from deepseek_moe_model import MODEL_NAME

torch.manual_seed(1111)

FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, base_config="E2M1_amax_E4M3"),
    "mxfp4": dict(block_size=32, base_config="E2M1_mxfp4_ue8m0"),
}
PROBE_SEQ_LEN = 2048

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# module-name suffix -> category; only one of q/k/v (shared input) and one of gate/up
CATEGORY_BY_SUFFIX = {
    "q_proj": "qkv_proj",
    "o_proj": "o_proj",
    "gate_proj": "gateup_proj",
    "down_proj": "down_proj",
}


@torch.no_grad()
def topk_error_ratio_of(x: torch.Tensor, block_size: int, base_config: str, k_values: list) -> dict:
    K = x.shape[-1]
    num_blocks = K // block_size
    if num_blocks == 0:
        return None
    x = x.reshape(-1, K)[:, : num_blocks * block_size].to(torch.float32)
    xq = quantize_tensor_config(x, base_config, block_size)
    eps = (x - xq).reshape(-1, block_size)
    eps_sq = eps.pow(2)
    eps_sq_sorted, _ = eps_sq.sort(dim=-1, descending=True)
    cum = eps_sq_sorted.cumsum(dim=-1)
    total_raw = cum[..., -1:]
    valid = (total_raw.squeeze(-1) > 0)
    if valid.sum() == 0:
        return None
    total = total_raw.clamp(min=1e-30)
    ratio_topk = (100 * cum / total)[valid].mean(dim=0)
    return {k: ratio_topk[k - 1].item() for k in k_values}


def accumulate(cat_sum: dict, cat_n: dict, category: str, res: dict, k_values: list):
    if res is None:
        return
    if category not in cat_sum:
        cat_sum[category] = {k: 0.0 for k in [0] + k_values}
        cat_n[category] = 0
    for k in k_values:
        cat_sum[category][k] += res[k]
    cat_n[category] += 1


@torch.no_grad()
def measure_model(fmt: str) -> dict:
    cfg = FORMAT_CONFIG[fmt]
    block_size, base_config = cfg["block_size"], cfg["base_config"]
    k_values = list(range(1, block_size + 1))

    same_seeds()
    full_text = load_wikitext2_text()
    _, encodings = load_tokenizer_and_encodings(MODEL_NAME, full_text)

    cat_sum, cat_n = {}, {}

    def probe(category):
        def fn(x):
            accumulate(cat_sum, cat_n, category, topk_error_ratio_of(x.detach(), block_size, base_config, k_values), k_values)
            return x
        return fn

    dsm.set_hooks(q=probe("Q"), k=probe("K"))
    model = dsm.load_model()

    handles = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or name == "lm_head":
            continue
        category = CATEGORY_BY_SUFFIX.get(name.rsplit(".", 1)[-1])
        if category is None:
            continue
        p = probe(category)
        handles.append(mod.register_forward_pre_hook(lambda _m, inputs, p=p: p(inputs[0])))
    print(f"Hooking {len(handles)} Linear modules (all experts' gate/down + q/o_proj)", flush=True)

    ids = encodings.input_ids[:, :PROBE_SEQ_LEN].to(model.device)
    model(input_ids=ids)
    dsm.set_hooks()
    for h in handles:
        h.remove()

    by_category = {}
    for category in sorted(cat_sum.keys()):
        avg = {0: 0.0}
        avg.update({k: cat_sum[category][k] / cat_n[category] for k in k_values})
        print(f"[{fmt}][deepseek_moe_16b] {category} ({cat_n[category]}): " + " ".join(f"top{k}={avg[k]:.2f}%" for k in k_values), flush=True)
        by_category[category] = avg

    del model
    cleanup_cuda()
    return by_category


def main():
    fmt = sys.argv[1]
    k_values = list(range(1, FORMAT_CONFIG[fmt]["block_size"] + 1))
    by_category = measure_model(fmt)

    csv_path = os.path.join(RESULTS_DIR, f"real_summary_{fmt}_deepseek_moe_16b.csv")
    categories = sorted(by_category.keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "k"] + categories)
        w.writeheader()
        for k in [0] + k_values:
            row = {"model": "deepseek_moe_16b", "k": k}
            for category in categories:
                row[category] = by_category[category].get(k, "")
            w.writerow(row)
    print(f"Saved -> {csv_path}", flush=True)
    print("H3_DEEPSEEK_MOE_DONE", flush=True)


if __name__ == "__main__":
    main()
