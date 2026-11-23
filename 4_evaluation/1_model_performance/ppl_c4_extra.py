"""C4 PPL for one arm outside ppl_c4.ARMS (mxfp4plus, m2xfp), same protocol as ppl_c4.py (256 x 2048-token
C4 samples, ppl_w4a4's build_model). Writes its own CSV seeded with the BF16 row so parallel runs don't clobber.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 ppl_c4_extra.py <fmt> <model> <arm>
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppl_c4 as C  # noqa: E402  (sets up sys.path for ppl_w4a4 / c4_utils / eval_utils)
from ppl_c4 import W, same_seeds, cleanup_cuda, load_tokenizer_and_encodings, measure_ppl, c4_encodings, N_SAMPLES, SEQ_LEN  # noqa: E402

LABEL = {"bf16": "BF16", "baseline": "Native_FP4", "mxfp6": "Full_Boost", "mxfp4plus": "MXFP4plus", "m2xfp": "M2XFP", "czardas_top2": "Czardas_top2", "magnitude_top2": "Magnitude_top2"}
FIELDS = ["format", "model", "setting", "ppl", "increase", "elapsed_sec"]


def main():
    fmt, model_key, arm = sys.argv[1:4]
    src = os.path.join(C.RESULTS_DIR, f"ppl_c4_{fmt}_{model_key}.csv")
    with open(src) as f:
        bf16_row = next(r for r in csv.DictReader(f) if r["setting"] == "BF16")
    bf16 = float(bf16_row["ppl"])

    same_seeds()
    tok, _ = load_tokenizer_and_encodings(W.model_name_of(model_key), "x")
    encodings = c4_encodings(tok)
    cache = W.make_cache_provider(fmt, model_key)(W.cache_kind(arm))
    model, restore = W.build_model(model_key, arm, fmt, cache)
    t0 = time.time()
    try:
        ppl = measure_ppl(model, encodings, f"{model_key}__c4__{fmt}__{LABEL[arm]}", model.device,
                          seq_len=SEQ_LEN, stride=SEQ_LEN, max_steps=N_SAMPLES)
    finally:
        restore()
    elapsed = time.time() - t0
    del model
    cleanup_cuda()
    print(f"[c4][{fmt}][{model_key}] {LABEL[arm]} PPL = {ppl:.6f}  ({ppl - bf16:+.4f} vs BF16)  elapsed={elapsed:.1f}s",
          flush=True)
    out = os.path.join(C.RESULTS_DIR, f"ppl_c4_{fmt}_{model_key}_{arm}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerow(bf16_row)
        w.writerow({"format": fmt, "model": model_key, "setting": LABEL[arm], "ppl": ppl,
                    "increase": ppl - bf16, "elapsed_sec": round(elapsed, 1)})
    print(f"Saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
