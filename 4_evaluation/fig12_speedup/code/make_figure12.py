"""Figure 12: speedup of Native FP4 and Crescent over All E2M3, per model and geometric mean (GM).

Reads results/format_comparison_token_mean.csv (inference-time increase over the family's 4-bit format,
averaged over 2,048 / 4,096 / 8,192 tokens; written by runpod_5090/assemble_5090.sh):
  speedup(Native)   = 1 + All_E2M3_increase
  speedup(Crescent) = (1 + All_E2M3_increase) / (1 + Crescent_increase)
Output: results/figure12_speedup.csv
"""
import csv
import math
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
MODELS = ["Llama", "Qwen", "DeepSeek"]
FAMILIES = [("NVFP4", "crescent_nv", "nvfp6"), ("MXFP4", "crescent_mx", "mxfp6")]


def main():
    pct = {(r["model"], r["metric"]): float(r["mean_pct"]) / 100
           for r in csv.DictReader(open(os.path.join(R, "format_comparison_token_mean.csv")))}
    rows = []
    for family, crescent, all_e2m3 in FAMILIES:
        nat, cre = [], []
        for m in MODELS:
            n = 1 + pct[(m, all_e2m3)]
            c = n / (1 + pct[(m, crescent)])
            nat.append(n)
            cre.append(c)
            rows.append(dict(family=family, model=m, native_fp4=f"{n:.3f}", crescent=f"{c:.3f}", all_e2m3="1.000"))
        gm = lambda v: math.exp(sum(map(math.log, v)) / len(v))
        rows.append(dict(family=family, model="GM", native_fp4=f"{gm(nat):.3f}", crescent=f"{gm(cre):.3f}", all_e2m3="1.000"))
    path = os.path.join(R, "figure12_speedup.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "model", "native_fp4", "crescent", "all_e2m3"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
