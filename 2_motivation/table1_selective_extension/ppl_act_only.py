"""Activation-only PPL (WikiText2): weights stay BF16 (empty GPTQ cache -> load_cache_into copies nothing),
activations quantized exactly as the given W4A4 arm (baseline = Native FP4 on Linear/Q/K/V/P with the current
causal-full P; czardas_top2 / magnitude_top2 = final config). Pair with weight_quantization/results/
weight_gptq_ppl_*.csv (weight-only GPTQ, BF16 activations).
Usage: python3 ppl_act_only.py <fmt> <model> <arm>"""
import csv, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import ppl_w4a4 as W
fmt, model_key, arm = sys.argv[1:4]
W.LABELS[arm] = W.LABELS[arm] + "_ActOnly"
W.ALL_ARMS = [arm]
src = os.path.join(W.RESULTS_DIR, f"ppl_{fmt}_{model_key}.csv")
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"ppl_{fmt}_{model_key}_actonly_{arm}.csv")
if not os.path.exists(dst):
    with open(src) as f:
        rows = [r for r in csv.DictReader(f) if r["setting"] == "BF16"]
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["format", "model", "setting", "ppl", "increase", "elapsed_sec"])
        w.writeheader(); w.writerows(rows)
W.same_seeds()
_, enc = W.load_tokenizer_and_encodings(W.model_name_of(model_key), W.load_wikitext2_text())
W.run_all(fmt, model_key, lambda kind: {}, enc, dst)
