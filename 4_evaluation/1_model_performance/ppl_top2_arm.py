"""WikiText2 PPL for one of the final-config arms (czardas_top2 / magnitude_top2, see ppl_w4a4.arm_setup).
Writes results/ppl_{fmt}_{model}_{czardas|magnitude}_top2.csv seeded with the BF16 row (same file
ppl_czardas_top2.py writes for Llama). Usage: python3 ppl_top2_arm.py <fmt> <model> <czardas_top2|magnitude_top2>"""
import csv, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import ppl_w4a4 as W
fmt, model_key, arm = sys.argv[1:4]
W.ALL_ARMS = [arm]
src = os.path.join(W.RESULTS_DIR, f"ppl_{fmt}_{model_key}.csv")
dst = os.path.join(W.RESULTS_DIR, f"ppl_{fmt}_{model_key}_{arm}.csv")
if not os.path.exists(dst):
    with open(src) as f:
        rows = [r for r in csv.DictReader(f) if r["setting"] == "BF16"]
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["format", "model", "setting", "ppl", "increase", "elapsed_sec"])
        w.writeheader(); w.writerows(rows)
W.same_seeds()
_, enc = W.load_tokenizer_and_encodings(W.model_name_of(model_key), W.load_wikitext2_text())
W.run_all(fmt, model_key, W.make_cache_provider(fmt, model_key), enc, dst)
