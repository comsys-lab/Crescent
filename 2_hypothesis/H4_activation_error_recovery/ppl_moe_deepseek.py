"""H4 (PPL), DeepSeek-MoE-16B-Base variant of ppl_moe_qwen3.py: BF16 vs plain NVFP4/MXFP4
(Linear+Q+K) vs EXACT top-k E2M7 boost, with the error-compensation decomposition for QK^T.
Model = common/deepseek_moe_model.py (pure-PyTorch port); experts are plain nn.Linear so
replace_linear_modules covers routed + shared experts.

Usage: CUDA_VISIBLE_DEVICES=<free GPUs> python3 ppl_moe_deepseek.py <nvfp4|mxfp4>
"""
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_tokenizer_and_encodings, measure_ppl
from formats import quantize_tensor_config, quantize_tensor_topk_activation_e2m7
from quant_targets import replace_linear_modules
import deepseek_moe_model as dsm
from deepseek_moe_model import MODEL_NAME

torch.manual_seed(1111)

FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, base_config="E2M1_amax_E4M3", boost_config="E2M7_amax_E4M3", k=2),
    "mxfp4": dict(block_size=32, base_config="E2M1_mxfp4_ue8m0", boost_config="E2M7_mxfp4_ue8m0", k=4),
}
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def identity(x):
    return x


def run_arm(arm, linear_fn, base_fns, boost_fns, encodings) -> float:
    """boost_fns None -> plain single-precision Q/K (base_fns["q"/"k"]); otherwise the
    error-compensation decomposition. Every Linear (attention, routed + shared experts,
    dense layer 0) is wrapped with linear_fn."""
    same_seeds()
    if boost_fns is None:
        dsm.set_hooks(q=base_fns["q"], k=base_fns["k"])
    else:
        dsm.set_decomposed(base_fns, boost_fns)
    model = dsm.load_model()
    n_linear = replace_linear_modules(model, linear_fn)
    print(f"[deepseek_moe_16b] {arm} wrapped {n_linear} linear layers", flush=True)
    t0 = time.time()
    try:
        ppl = measure_ppl(model, encodings, f"deepseek_moe_16b__{arm}", model.device)
    finally:
        dsm.set_hooks()
        dsm.set_decomposed()
    elapsed = time.time() - t0
    print(f"[deepseek_moe_16b] {arm} PPL = {ppl}  elapsed={elapsed:.1f}s", flush=True)
    del model
    cleanup_cuda()
    return ppl, elapsed


def bf16_ppl(encodings):
    same_seeds()
    model = dsm.load_model()
    t0 = time.time()
    ppl = measure_ppl(model, encodings, "deepseek_moe_16b__BF16", model.device)
    elapsed = time.time() - t0
    print(f"[deepseek_moe_16b] BF16 PPL = {ppl}  elapsed={elapsed:.1f}s", flush=True)
    del model
    cleanup_cuda()
    return ppl, elapsed


def save(csv_path, fields, rows):
    # after every arm: a late-arm crash must not discard earlier arms
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Saved -> {csv_path}", flush=True)


def main():
    fmt = sys.argv[1]
    cfg = FORMAT_CONFIG[fmt]
    block_size, base_config, boost_config, k = cfg["block_size"], cfg["base_config"], cfg["boost_config"], cfg["k"]
    same_seeds()
    _, encodings = load_tokenizer_and_encodings(MODEL_NAME, load_wikitext2_text())
    csv_path = os.path.join(RESULTS_DIR, f"ppl_{fmt}_deepseek_moe_16b.csv")
    fields = ["model", "arm", "k", "ppl", "elapsed_sec"]
    rows = []

    def add(arm, kk, res):
        rows.append({"model": "deepseek_moe_16b", "arm": arm, "k": kk, "ppl": res[0], "elapsed_sec": round(res[1], 3)})
        save(csv_path, fields, rows)

    def plain_fn(x):
        return quantize_tensor_config(x.detach().to(torch.float32), base_config, block_size).to(x.dtype)

    def topk_fn(x):
        return quantize_tensor_topk_activation_e2m7(x, k, block_size=block_size,
                                                      base_config=base_config, boost_config=boost_config).to(x.dtype)

    add("BF16", "", bf16_ppl(encodings))
    add(f"{fmt.upper()}_ALL", "", run_arm(f"{fmt.upper()}_ALL", plain_fn,
                                          {"q": plain_fn, "k": plain_fn}, None, encodings))
    base_fns = {"q": plain_fn, "k": plain_fn, "v": identity, "p": identity}
    boost_fns = {"q": topk_fn, "k": topk_fn, "v": identity, "p": identity}
    add(f"E2M7_AT_TOP{k}_ALL", k, run_arm(f"E2M7_AT_TOP{k}_ALL", topk_fn, base_fns, boost_fns, encodings))
    print("H4_DEEPSEEK_MOE_PPL_DONE", flush=True)


if __name__ == "__main__":
    main()
