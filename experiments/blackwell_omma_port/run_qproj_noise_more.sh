#!/usr/bin/env bash
# 14 more launch-latency perturbations of the q_proj 12op/widened pair (20 in total
# with run_qproj_studies.sh), at most MAX_SIMS accel-sim processes at once.
set -uo pipefail
cd "$(dirname "$0")"
MAX_SIMS=${MAX_SIMS:-22}
T=traces/m3tie_linear_2048x4096x4096
for L in 10 13 17 21 26 31 37 42 50 61 70 83 97 111; do
  while [ "$(pgrep -xc accel-sim.out)" -ge $((MAX_SIMS - 1)) ]; do sleep 30; done
  for m in 12op widened; do
    ./run_sim.sh ${T}_$m qp_noise_L${L}_$m -gpgpu_kernel_launch_latency $L > /dev/null 2>&1 &
  done
  sleep 5
done
wait
echo "$(date '+%F %T') done"
