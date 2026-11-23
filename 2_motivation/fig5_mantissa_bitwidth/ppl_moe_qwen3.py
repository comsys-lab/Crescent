"""H4 (v4, PPL), Qwen3-30B-A3B MoE variant -- PPL at boost mantissa=3,
mirroring ppl.py's methodology (see H3/ppl_moe_qwen3.py's module docstring
for the Q/K/V/O_proj vs gateup_proj/down_proj split rationale -- identical
here). M=1 (=H3 MoE's *_ALL) and M=7 (=H3 MoE's E2M7_AT_TOPk_ALL) are
already covered by H3's ppl_moe_qwen3.py; only M=3 is new.

Usage: CUDA_VISIBLE_DEVICES=<free GPUs only> python3 ppl_moe_qwen3.py <nvfp4|mxfp4>
"""
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_tokenizer_and_encodings, measure_ppl
from formats import quantize_tensor_config, quantize_tensor_topk_activation_e2m7
from quant_targets import replace_linear_modules
from qwen3_moe_targets import MODEL_NAME, patch_eager_qwen3_moe_decomposed, patch_qwen3_moe_experts

torch.manual_seed(1111)

MANTISSA_VALUES = [3]

FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, base_config="E2M1_amax_E4M3", k=2),
    "mxfp4": dict(block_size=32, base_config="E2M1_mxfp4_ue8m0", k=4),
}
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def identity(x):
    return x


def boost_config_name(fmt: str, mant_bits: int) -> str:
    return f"E2M{mant_bits}_amax_E4M3" if fmt == "nvfp4" else f"E2M{mant_bits}_mxfp4_ue8m0"


def measure(fmt: str) -> list:
    cfg = FORMAT_CONFIG[fmt]
    k, block_size, base_config = cfg["k"], cfg["block_size"], cfg["base_config"]

    same_seeds()
    full_text = load_wikitext2_text()
    _, encodings = load_tokenizer_and_encodings(MODEL_NAME, full_text)

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

        import transformers.models.qwen3_moe.modeling_qwen3_moe as attn_mod
        attn_cls, orig_attn_forward = patch_eager_qwen3_moe_decomposed(attn_mod, base_fns, boost_fns)
        experts_cls, orig_experts_forward = patch_qwen3_moe_experts(attn_mod, quant_fn, quant_fn)

        from transformers import AutoModelForCausalLM
        same_seeds()
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                                       low_cpu_mem_usage=True, attn_implementation="eager",
                                                       device_map="auto")
        model.eval()
        n_linear = replace_linear_modules(model, quant_fn)
        print(f"[qwen3_moe] {arm_name} wrapped {n_linear} linear layers", flush=True)
        t0 = time.time()
        try:
            ppl = measure_ppl(model, encodings, f"qwen3_moe__{arm_name}", model.device)
        finally:
            attn_cls.forward = orig_attn_forward
            experts_cls.forward = orig_experts_forward
        elapsed = time.time() - t0
        print(f"[qwen3_moe] {arm_name} PPL = {ppl}  elapsed={elapsed:.1f}s", flush=True)
        rows.append({"format": fmt, "model": "qwen3_moe", "k": k, "boost_mant_bits": m,
                     "ppl": ppl, "elapsed_sec": round(elapsed, 3)})
        del model
        cleanup_cuda()

    return rows


def main():
    fmt = sys.argv[1]
    rows = measure(fmt)
    csv_path = os.path.join(RESULTS_DIR, f"ppl_{fmt}_qwen3_moe.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["format", "model", "k", "boost_mant_bits", "ppl", "elapsed_sec"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Saved -> {csv_path}", flush=True)
    print("H4_QWEN3_MOE_PPL_DONE", flush=True)


if __name__ == "__main__":
    main()
