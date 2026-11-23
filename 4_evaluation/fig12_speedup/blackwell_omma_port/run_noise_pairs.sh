#!/usr/bin/env bash
# 20 launch-latency perturbations of a 12op/widened trace pair; paired pct statistics.
# Usage: run_noise_pairs.sh <trace prefix, e.g. traces/all_llama_qkt_M8192> <tag>
# Results: logs_noise_<tag>.csv and a summary line in logs_noise_<tag>.txt
set -uo pipefail
cd "$(dirname "$0")"
T=$1; TAG=$2; MAX_SIMS=${MAX_SIMS:-24}
LS="0 1 2 3 5 8 10 13 17 21 26 31 37 42 50 61 70 83 97 111"
for L in $LS; do
  while [ "$(pgrep -xc accel-sim.out)" -ge $((MAX_SIMS - 1)) ]; do sleep 30; done
  for m in 12op widened; do
    ./run_sim.sh ${T}_$m ${TAG}_L${L}_$m -gpgpu_kernel_launch_latency $L > /dev/null 2>&1 &
  done
  sleep 5
done
wait
python3 - "$TAG" $LS <<'PY'
import sys, statistics as st
tag, Ls = sys.argv[1], sys.argv[2:]
def g(L, m):
    for l in open(f"sim_runs/{tag}_L{L}_{m}/sim.log"):
        if l.startswith("gpu_tot_sim_cycle"): return int(l.split()[2])
rows = [(int(L), g(L, "12op"), g(L, "widened")) for L in Ls]
with open(f"logs_noise_{tag}.csv", "w") as f:
    f.write("launch_latency,cycles_12op,cycles_widened,pct\n")
    for L, a, b in rows: f.write(f"{L},{a},{b},{100*(b-a)/a:.4f}\n")
p = [100*(b-a)/a for _, a, b in rows]; n = len(p); m = st.mean(p); se = st.stdev(p)/n**0.5
print(f"RESULT {tag} n={n} mean={m:+.3f}% sd={st.stdev(p):.3f}% se={se:.3f}% 95%CI=[{m-2.09*se:+.3f}%, {m+2.09*se:+.3f}%]")
PY
