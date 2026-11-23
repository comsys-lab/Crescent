"""H4 (v4, PPL): PPL at boost mantissa=3 (the value real.py's sweep shows
saturates most of the recovery), applied to Linear, Q and K
independently via exact top-k, k fixed at 12.5% of block uniformly
(matching H2/H5). V and P stay BF16 always (dropped from the target
set -- H0 isolation showed negligible PPL impact everywhere). M=1 (no
boost, = H3's *_ALL plain arm) and M=7 (= H3's E2M7_AT_TOP{k}_ALL) are
already covered by H3 -- only M=3 is new here.

QK^T uses the error-compensation decomposition (drops the delta x delta
cross term when Q/K happen to be boosted at the same position) -- see
H3/ppl.py's module docstring. P@V needs no such handling (both stay
BF16, never quantized, so that matmul is exact).

Usage: python3 ppl.py <nvfp4|mxfp4> [model ...]
"""

import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_tokenizer_and_encodings, measure_ppl
from formats import quantize_tensor_config, quantize_tensor_topk_activation_e2m7
from quant_targets import replace_linear_modules, patch_eager_decomposed, MODEL_CONFIG

torch.manual_seed(1111)

MANTISSA_VALUES = [3]

FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, base_config="E2M1_amax_E4M3", k=2),
    "mxfp4": dict(block_size=32, base_config="E2M1_mxfp4_ue8m0", k=4),
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def boost_config_name(fmt: str, mant_bits: int) -> str:
    return f"E2M{mant_bits}_amax_E4M3" if fmt == "nvfp4" else f"E2M{mant_bits}_mxfp4_ue8m0"


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


def measure(fmt: str, model_key: str) -> list:
    cfg = FORMAT_CONFIG[fmt]
    model_cfg = MODEL_CONFIG[model_key]
    model_name, arch = model_cfg["model_name"], model_cfg["arch"]
    k, block_size, base_config = cfg["k"], cfg["block_size"], cfg["base_config"]

    same_seeds()
    full_text = load_wikitext2_text()
    _, encodings = load_tokenizer_and_encodings(model_name, full_text)

    rows = []
    for m in MANTISSA_VALUES:
        boost_config = boost_config_name(fmt, m)
        arm_name = f"{fmt.upper()}_E2M{m}_AT_TOP{k}_ALL"

        def quant_fn(x, boost_config=boost_config):
            return quantize_tensor_topk_activation_e2m7(x, k, block_size=block_size,
                                                          base_config=base_config, boost_config=boost_config).to(x.dtype)

        def plain_fn(x):
            return quantize_tensor_config(x.detach().to(torch.float32), base_config, block_size).to(x.dtype)

        base_fns = {"q": plain_fn, "k": plain_fn, "v": identity, "p": identity}
        boost_fns = {"q": quant_fn, "k": quant_fn, "v": identity, "p": identity}

        from transformers import AutoModelForCausalLM
        same_seeds()
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                       low_cpu_mem_usage=True, attn_implementation="eager")
        model.to(DEVICE)
        model.eval()
        n_linear = replace_linear_modules(model, quant_fn)
        attn_mod = attn_mod_for(model_name, arch)
        attn_cls, orig_forward = patch_eager_decomposed(arch, attn_mod, base_fns, boost_fns)
        print(f"[{fmt}][{model_key}] {arm_name} wrapped {n_linear} linear layers", flush=True)
        t0 = time.time()
        try:
            ppl = measure_ppl(model, encodings, f"{fmt}_{model_key}__{arm_name}", DEVICE)
        finally:
            attn_cls.forward = orig_forward
        elapsed = time.time() - t0
        print(f"[{fmt}][{model_key}] {arm_name} PPL = {ppl}  elapsed={elapsed:.1f}s", flush=True)
        rows.append({"format": fmt, "model": model_key, "k": k, "boost_mant_bits": m,
                     "ppl": ppl, "elapsed_sec": round(elapsed, 3)})
        del model
        cleanup_cuda()

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
            w = csv.DictWriter(f, fieldnames=["format", "model", "k", "boost_mant_bits", "ppl", "elapsed_sec"])
            w.writeheader()
            for row in all_rows:
                w.writerow(row)
        print(f"Saved -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
