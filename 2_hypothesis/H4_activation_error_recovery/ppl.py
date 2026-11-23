"""H3 (v4): PPL for BF16 vs plain NVFP4/MXFP4 (Linear+Q+K, this
project's baseline -- V and P dropped, H0 isolation showed negligible
PPL impact everywhere) vs EXACT top-k boost (E2M7 at the top-k largest-
error positions per block, global sort) applied to Linear, Q and K
independently -- each surface picks its OWN top-k by its OWN error, no
cross-surface/joint selection. k fixed at 12.5% of block (matching
H2/H5's fixed ratio) uniformly across every surface.

QK^T (activation x activation matmul) uses the error-compensation
decomposition (patch_eager_decomposed): since Q's and K's
independently-chosen boost positions routinely overlap, naively
multiplying two boosted tensors would require a boosted x boosted
(e.g. FP4-mantissa-7 x FP4-mantissa-7) multiplier that the real hardware
correction datapath does not have -- it only ever multiplies a base-
precision value against a delta. The decomposition drops that delta x
delta cross term exactly, matching real hardware. P@V needs no such
handling any more -- P and V both stay BF16 (never quantized), so that
matmul is exact. Linear's activation x WEIGHT matmul needs no such
handling either (weight is exact BF16, never boosted).

Usage: python3 ppl.py <nvfp4|mxfp4> [model ...]
"""

import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_tokenizer_and_encodings, measure_ppl
from formats import quantize_tensor_config, quantize_tensor_topk_activation_e2m7
from quant_targets import replace_linear_modules, patch_eager, patch_eager_decomposed, MODEL_CONFIG

torch.manual_seed(1111)

FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, base_config="E2M1_amax_E4M3", boost_config="E2M7_amax_E4M3", k=2),
    "mxfp4": dict(block_size=32, base_config="E2M1_mxfp4_ue8m0", boost_config="E2M7_mxfp4_ue8m0", k=4),
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def identity(x):
    return x


def attn_mod_for(model_name: str, arch: str):
    if arch == "opt":
        import transformers.models.opt.modeling_opt as opt_mod
        return opt_mod
    if "ministral" in model_name.lower():
        import transformers.models.ministral.modeling_ministral as ministral_mod
        return ministral_mod
    if "mistral" in model_name.lower():
        import transformers.models.mistral.modeling_mistral as mistral_mod
        return mistral_mod
    import transformers.models.llama.modeling_llama as llama_mod
    return llama_mod


def measure_plain_arm(model_name: str, arch: str, arm: str, plain_fn, encodings) -> float:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                   low_cpu_mem_usage=True, attn_implementation="eager")
    model.to(DEVICE)
    model.eval()
    n_linear = replace_linear_modules(model, plain_fn)
    attn_mod = attn_mod_for(model_name, arch)
    attn_cls, orig_forward = patch_eager(arch, attn_mod, plain_fn, plain_fn, identity, identity)
    print(f"[{arm}] wrapped {n_linear} linear layers", flush=True)
    try:
        ppl = measure_ppl(model, encodings, arm, DEVICE)
    finally:
        attn_cls.forward = orig_forward
    del model
    cleanup_cuda()
    return ppl


def measure_boosted_arm(model_name: str, arch: str, arm: str, linear_boost_fn, base_fns, boost_fns, encodings) -> float:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                   low_cpu_mem_usage=True, attn_implementation="eager")
    model.to(DEVICE)
    model.eval()
    n_linear = replace_linear_modules(model, linear_boost_fn)
    attn_mod = attn_mod_for(model_name, arch)
    attn_cls, orig_forward = patch_eager_decomposed(arch, attn_mod, base_fns, boost_fns)
    print(f"[{arm}] wrapped {n_linear} linear layers", flush=True)
    try:
        ppl = measure_ppl(model, encodings, arm, DEVICE)
    finally:
        attn_cls.forward = orig_forward
    del model
    cleanup_cuda()
    return ppl


def measure(fmt: str, model_key: str) -> list:
    cfg = FORMAT_CONFIG[fmt]
    block_size, base_config, boost_config, k = cfg["block_size"], cfg["base_config"], cfg["boost_config"], cfg["k"]
    model_cfg = MODEL_CONFIG[model_key]
    model_name, arch = model_cfg["model_name"], model_cfg["arch"]

    same_seeds()
    full_text = load_wikitext2_text()
    _, encodings = load_tokenizer_and_encodings(model_name, full_text)

    rows = []

    from transformers import AutoModelForCausalLM
    same_seeds()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to(DEVICE); model.eval()
    t0 = time.time()
    ppl = measure_ppl(model, encodings, f"{model_key}__BF16", DEVICE)
    elapsed = time.time() - t0
    print(f"[{model_key}] BF16 PPL = {ppl}  elapsed={elapsed:.1f}s", flush=True)
    rows.append({"model": model_key, "arm": "BF16", "k": "", "ppl": ppl, "elapsed_sec": round(elapsed, 3)})
    del model
    cleanup_cuda()

    def plain_fn(x):
        return quantize_tensor_config(x.detach().to(torch.float32), base_config, block_size).to(x.dtype)

    same_seeds()
    t0 = time.time()
    ppl = measure_plain_arm(model_name, arch, f"{model_key}__{fmt.upper()}_ALL", plain_fn, encodings)
    elapsed = time.time() - t0
    print(f"[{model_key}] {fmt.upper()}_ALL (Linear+Q+K plain) PPL = {ppl}  elapsed={elapsed:.1f}s", flush=True)
    rows.append({"model": model_key, "arm": f"{fmt.upper()}_ALL", "k": "", "ppl": ppl, "elapsed_sec": round(elapsed, 3)})

    arm_name = f"E2M7_AT_TOP{k}_ALL"

    def topk_fn(x):
        return quantize_tensor_topk_activation_e2m7(x, k, block_size=block_size,
                                                      base_config=base_config, boost_config=boost_config).to(x.dtype)

    base_fns = {"q": plain_fn, "k": plain_fn, "v": identity, "p": identity}
    boost_fns = {"q": topk_fn, "k": topk_fn, "v": identity, "p": identity}

    same_seeds()
    t0 = time.time()
    ppl = measure_boosted_arm(model_name, arch, f"{model_key}__{arm_name}", topk_fn, base_fns, boost_fns, encodings)
    elapsed = time.time() - t0
    print(f"[{model_key}] {arm_name} PPL = {ppl}  elapsed={elapsed:.1f}s", flush=True)
    rows.append({"model": model_key, "arm": arm_name, "k": k, "ppl": ppl, "elapsed_sec": round(elapsed, 3)})

    return rows


def main():
    fmt = sys.argv[1]
    model_keys = sys.argv[2:] if len(sys.argv) > 2 else ["llama_base"]
    all_rows = []
    suffix = os.environ.get("RESULT_SUFFIX", "")
    if len(model_keys) == 1:  # keep the model in the file name (e.g. ppl_nvfp4_llama_base.csv)
        suffix = f"_{model_keys[0]}{suffix}"
    csv_path = os.path.join(RESULTS_DIR, f"ppl_{fmt}{suffix}.csv")
    for mk in model_keys:
        all_rows.extend(measure(fmt, mk))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "arm", "k", "ppl", "elapsed_sec"])
            w.writeheader()
            for row in all_rows:
                w.writerow(row)
        print(f"Saved -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
