"""H3 (v4), Qwen3-30B-A3B MoE variant -- PPL for BF16 vs plain NVFP4/MXFP4
vs EXACT top-k boost (E2M7), mirroring ppl.py's methodology:
  - Q/K/V/O_proj: plain nn.Linear, quant_targets.replace_linear_modules
    handles the Linear-input (qkv_proj/o_proj) categories unchanged.
  - gateup_proj/down_proj (MoE experts): no nn.Linear to wrap -- routed
    through patch_qwen3_moe_experts instead (single-sided quantize, no
    decomposition needed -- expert weights are exact BF16, see
    common/qwen3_moe_targets.py's module docstring).
  - Q/K (attention tensors): patch_eager_qwen3_moe_decomposed, same
    error-compensation decomposition as the dense models' QK^T (drops the
    delta x delta cross term) -- see H3/ppl.py's module docstring for the
    derivation. P@V needs no such handling (both stay BF16).

Patches the attention/experts classes BEFORE from_pretrained(device_map=
"auto") in every arm -- accelerate's dispatch_model captures each module's
forward at load time, patching after never takes effect (see HANDOFF_MOE_
UPGRADE.md's device_map gotcha).

Usage: CUDA_VISIBLE_DEVICES=<free GPUs only> python3 ppl_moe_qwen3.py <nvfp4|mxfp4>
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
from qwen3_moe_targets import MODEL_NAME, patch_eager_qwen3_moe, patch_eager_qwen3_moe_decomposed, patch_qwen3_moe_experts

torch.manual_seed(1111)

FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, base_config="E2M1_amax_E4M3", boost_config="E2M7_amax_E4M3", k=2),
    "mxfp4": dict(block_size=32, base_config="E2M1_mxfp4_ue8m0", boost_config="E2M7_mxfp4_ue8m0", k=4),
}
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def identity(x):
    return x


def load_moe_model(attn_patch_fn, experts_patch_fn):
    """Both class patches are applied BEFORE from_pretrained so accelerate's device-placement
    hooks capture them. (An earlier version patched the experts AFTER loading, which silently
    never took effect: expert inputs stayed BF16 in the ALL / boost arms -- verified by a probe
    that got 0 calls.)"""
    import transformers.models.qwen3_moe.modeling_qwen3_moe as attn_mod
    attn_cls, orig_attn_forward = attn_patch_fn(attn_mod)
    experts_cls, orig_experts_forward = experts_patch_fn(attn_mod)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                                   low_cpu_mem_usage=True, attn_implementation="eager",
                                                   device_map="auto")
    model.eval()
    return model, attn_cls, orig_attn_forward, experts_cls, orig_experts_forward


def measure(fmt: str) -> list:
    cfg = FORMAT_CONFIG[fmt]
    block_size, base_config, boost_config, k = cfg["block_size"], cfg["base_config"], cfg["boost_config"], cfg["k"]

    same_seeds()
    full_text = load_wikitext2_text()
    _, encodings = load_tokenizer_and_encodings(MODEL_NAME, full_text)

    rows = []

    # ---- BF16 ----
    from transformers import AutoModelForCausalLM
    same_seeds()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                                   low_cpu_mem_usage=True, device_map="auto")
    model.eval()
    t0 = time.time()
    ppl = measure_ppl(model, encodings, "qwen3_moe__BF16", model.device)
    elapsed = time.time() - t0
    print(f"[qwen3_moe] BF16 PPL = {ppl}  elapsed={elapsed:.1f}s", flush=True)
    rows.append({"model": "qwen3_moe", "arm": "BF16", "k": "", "ppl": ppl, "elapsed_sec": round(elapsed, 3)})
    del model
    cleanup_cuda()

    def plain_fn(x):
        return quantize_tensor_config(x.detach().to(torch.float32), base_config, block_size).to(x.dtype)

    def topk_fn(x):
        return quantize_tensor_topk_activation_e2m7(x, k, block_size=block_size,
                                                      base_config=base_config, boost_config=boost_config).to(x.dtype)

    # ---- ALL (plain, single precision level, no decomposition needed) ----
    same_seeds()
    model, attn_cls, orig_attn_forward, experts_cls, orig_experts_forward = load_moe_model(
        lambda mod: patch_eager_qwen3_moe(mod, plain_fn, plain_fn, identity, identity),
        lambda mod: patch_qwen3_moe_experts(mod, plain_fn, plain_fn))
    n_linear = replace_linear_modules(model, plain_fn)
    arm = f"{fmt.upper()}_ALL"
    print(f"[qwen3_moe] {arm} wrapped {n_linear} linear layers", flush=True)
    t0 = time.time()
    try:
        ppl = measure_ppl(model, encodings, f"qwen3_moe__{arm}", model.device)
    finally:
        attn_cls.forward = orig_attn_forward
        experts_cls.forward = orig_experts_forward
    elapsed = time.time() - t0
    print(f"[qwen3_moe] {arm} PPL = {ppl}  elapsed={elapsed:.1f}s", flush=True)
    rows.append({"model": "qwen3_moe", "arm": arm, "k": "", "ppl": ppl, "elapsed_sec": round(elapsed, 3)})
    del model
    cleanup_cuda()

    # ---- E2M7 exact top-k boost, decomposed QK^T ----
    base_fns = {"q": plain_fn, "k": plain_fn, "v": identity, "p": identity}
    boost_fns = {"q": topk_fn, "k": topk_fn, "v": identity, "p": identity}
    arm = f"E2M7_AT_TOP{k}_ALL"

    same_seeds()
    model, attn_cls, orig_attn_forward, experts_cls, orig_experts_forward = load_moe_model(
        lambda mod: patch_eager_qwen3_moe_decomposed(mod, base_fns, boost_fns),
        lambda mod: patch_qwen3_moe_experts(mod, topk_fn, topk_fn))
    n_linear = replace_linear_modules(model, topk_fn)
    print(f"[qwen3_moe] {arm} wrapped {n_linear} linear layers", flush=True)
    t0 = time.time()
    try:
        ppl = measure_ppl(model, encodings, f"qwen3_moe__{arm}", model.device)
    finally:
        attn_cls.forward = orig_attn_forward
        experts_cls.forward = orig_experts_forward
    elapsed = time.time() - t0
    print(f"[qwen3_moe] {arm} PPL = {ppl}  elapsed={elapsed:.1f}s", flush=True)
    rows.append({"model": "qwen3_moe", "arm": arm, "k": k, "ppl": ppl, "elapsed_sec": round(elapsed, 3)})
    del model
    cleanup_cuda()

    return rows


def main():
    fmt = sys.argv[1]
    rows = measure(fmt)
    csv_path = os.path.join(RESULTS_DIR, f"ppl_{fmt}_qwen3_moe.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "arm", "k", "ppl", "elapsed_sec"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Saved -> {csv_path}", flush=True)
    print("H3_QWEN3_MOE_PPL_DONE", flush=True)


if __name__ == "__main__":
    main()
