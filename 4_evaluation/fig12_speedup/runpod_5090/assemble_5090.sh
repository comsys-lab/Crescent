#!/usr/bin/env bash
# Rebuild the execution-time tables from the RTX 5090 measurements (results/ copied back from RunPod).
# Usage: assemble_5090.sh <runpod results dir>
#  - overlays every file the RunPod run re-measured onto 4_evaluation/fig12_speedup/{results,log} (in place),
#  - kept from the RTX PRO 4000 runs: Accel-Sim operand fractions (results/accel_sim/*.csv, option (a)),
#    and the old HF-eager Qwen/DeepSeek forward files read by assemble_final_table.py (they only enter as
#    old/denominator x denominator/old and cancel), plus static per-model layer counts,
#  - the operand-term GEMM weight of the MoE experts uses the re-measured grouped baseline (grp_base_p0)
#    instead of the old grouped_raw.txt,
#  - runs assemble_format_comparison.py and writes the token-mean CSV.
set -euo pipefail
SRC=$(cd "$1" && pwd)
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
DST=$ROOT/4_evaluation/fig12_speedup
mkdir -p "$DST/log"
for f in gs8_16_raw.txt pp_raw.txt moe_quant_pp_raw.txt grouped_pp_raw.txt fc_dense.txt fc_quant.txt fc_grouped.txt moe_other.txt; do
  tail -1 "$SRC/$f" | grep -q '^DONE$' || { echo "incomplete: $f"; exit 1; }
  cp "$SRC/$f" "$DST/results/accel_sim/$f"; done
for f in qkt_pv_repeat.txt bf16_qkt_pv_repeat.txt; do
  tail -1 "$SRC/$f" | grep -q '^DONE$' || { echo "incomplete: $f"; exit 1; }
  cp "$SRC/$f" "$DST/results/$f"; done
for f in linear_stage_corrected_5rep.log bf16_real_attention_5rep.log; do
  tail -1 "$SRC/log/$f" | grep -q '^DONE$' || { echo "incomplete: log/$f"; exit 1; }
  cp "$SRC/log/$f" "$DST/log/$f"; done
cp "$SRC/gpu.txt" "$SRC/devprops.txt" "$DST/results/" 2>/dev/null || true
python3 - "$DST/code/assemble_final_table_gs8_16_pp.py" <<'PY'
import sys; p = sys.argv[1]; s = open(p).read()
old = 'lin_g += nl * (2 * ft.grp[u] + ft.grp[d])'
if old in s:  # already applied in the committed code/ (idempotent)
    open(p, "w").write(s.replace(old, 'lin_g += nl * (2 * G(u, "grp_base_p0") + G(d, "grp_base_p0"))  # 5090: re-measured grouped baseline'))
PY
cd "$DST/code"
python3 assemble_format_comparison.py
python3 - <<'PY'
import csv, statistics as st
rows = list(csv.DictReader(open("../results/format_comparison.csv")))
out = []
for m in ("Llama", "Qwen", "DeepSeek"):
    for k in ("crescent_nv", "nvfp6", "crescent_mx", "mxfp6"):
        v = [float(r[k]) for r in rows if r["model"] == m]
        out.append(dict(model=m, metric=k, mean_pct=f"{st.mean(v):.2f}", min_pct=f"{min(v):.2f}", max_pct=f"{max(v):.2f}"))
with open("../results/format_comparison_token_mean.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print("\nToken mean (inference-time increase vs the family's 4-bit format)")
print(f"{'Model':9s} {'Crescent(NVFP4)':>16s} {'All Boost(NVFP4)':>17s} {'Crescent(MXFP4)':>16s} {'All Boost(MXFP4)':>17s}")
for m in ("Llama", "Qwen", "DeepSeek"):
    d = {r["metric"]: r["mean_pct"] for r in out if r["model"] == m}
    print(f"{m:9s} {d['crescent_nv']+'%':>16s} {d['nvfp6']+'%':>17s} {d['crescent_mx']+'%':>16s} {d['mxfp6']+'%':>17s}")
PY
echo "wrote $DST/results/format_comparison.csv and format_comparison_token_mean.csv"
python3 make_figure12.py
