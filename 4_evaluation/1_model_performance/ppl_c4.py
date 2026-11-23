"""C4 PPL: BF16 vs Native FP4 (W4A4 baseline) vs Ours vs Magnitude (prior-work-style selection), on
256 x 2048-token C4 samples (standard GPTQ protocol, common/c4_utils.py). Reuses evaluation/ppl_w4a4.py's
model builder / GPTQ weight caches / arm setup UNCHANGED (only "bf16", "baseline", "grouped", "magnitude"
are run here -- the rotation ablations are out of scope for this comparison). Native_FP4/Ours/Magnitude
all quantize every activation operand feeding a matmul (Linear/Q/K/V/P -- real W4A4).

Usage: CUDA_VISIBLE_DEVICES=<GPUs> python3 ppl_c4.py <nvfp4|mxfp4> <llama_base|qwen3_moe|deepseek_moe_16b>
"""
import csv
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
HYP_ROOT = os.path.join(HERE, "..", "..", "2_hypothesis")
sys.path.insert(0, os.path.join(HYP_ROOT, "common"))
sys.path.insert(0, os.path.join(HYP_ROOT, "evaluation"))
from eval_utils import same_seeds, cleanup_cuda, load_tokenizer_and_encodings, measure_ppl  # noqa: E402
from c4_utils import c4_encodings, N_SAMPLES, SEQ_LEN  # noqa: E402
import ppl_w4a4 as W  # noqa: E402

torch.manual_seed(1111)

ARMS = ["bf16", "baseline", "grouped", "magnitude", "mxfp6", "fullrot"]  # + Full Boost, Rotation (QuaRot-style full Hadamard, no selective boost)
LABELS = {"bf16": "BF16", "baseline": "Native_FP4", "grouped": "Ours", "magnitude": "Magnitude", "mxfp6": "Full_Boost",
          "fullrot": "Rotation"}
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    fmt, model_key = sys.argv[1], sys.argv[2]
    csv_path = os.path.join(RESULTS_DIR, f"ppl_c4_{fmt}_{model_key}.csv")

    rows = []
    if os.path.exists(csv_path):  # resume: keep finished arms
        with open(csv_path) as f:
            rows = [dict(r, ppl=float(r["ppl"]), increase=float(r["increase"])) for r in csv.DictReader(f)]
    done = {r["setting"]: r["ppl"] for r in rows}
    bf16 = done.get(LABELS["bf16"])

    same_seeds()
    tok, _ = load_tokenizer_and_encodings(W.model_name_of(model_key), "x")  # "x": full_text unused, only need the tokenizer
    print(f"[c4][{model_key}] building {N_SAMPLES} x {SEQ_LEN}-token C4 samples (seed 0, GPTQ protocol)...", flush=True)
    encodings = c4_encodings(tok)
    print(f"[c4][{model_key}] input_ids shape {tuple(encodings.input_ids.shape)}", flush=True)

    cache_for = W.make_cache_provider(fmt, model_key)  # lazy per-kind cache (plain / fullrot), loads/builds on first use

    for arm in ARMS:
        label = LABELS[arm]
        if label in done:
            print(f"[c4][{fmt}][{model_key}] {label} already measured ({done[label]:.6f}) -- skipping", flush=True)
            continue
        if arm == "fullrot" and model_key == "qwen3_moe":
            sys.path.insert(0, os.path.join(HYP_ROOT, "weight_quantization"))
            from fullrot_cache import cache_path as _fullrot_cache_path
            if not os.path.exists(_fullrot_cache_path(fmt, model_key)):
                print(f"[c4][{fmt}][{model_key}] skipping {label}: Qwen fullrot GPTQ cache build needs ~118GB peak "
                      f"host RAM and isn't built for this format yet -- deferred, build separately with "
                      f"fullrot_cache.py on a clear machine", flush=True)
                continue
        same_seeds()
        cache = cache_for(W.cache_kind(arm)) if arm != "bf16" else None
        model, restore = W.build_model(model_key, arm, fmt, cache)
        t0 = time.time()
        try:
            ppl = measure_ppl(model, encodings, f"{model_key}__c4__{fmt}__{label}", model.device,
                              seq_len=SEQ_LEN, stride=SEQ_LEN, max_steps=N_SAMPLES)
        finally:
            restore()
        elapsed = time.time() - t0
        del model
        cleanup_cuda()
        if arm == "bf16":
            bf16 = ppl
        print(f"[c4][{fmt}][{model_key}] {label} PPL = {ppl:.6f}" + ("" if arm == "bf16" else f"  ({ppl - bf16:+.4f} vs BF16)")
              + f"  elapsed={elapsed:.1f}s", flush=True)
        rows.append({"format": fmt, "model": model_key, "setting": label, "ppl": ppl, "increase": ppl - bf16,
                     "elapsed_sec": round(elapsed, 1)})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["format", "model", "setting", "ppl", "increase", "elapsed_sec"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Saved -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
