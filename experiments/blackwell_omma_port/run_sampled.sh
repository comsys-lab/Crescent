#!/usr/bin/env bash
# Simulate only the first <waves> x 70 CTAs (-gpgpu_max_cta) of a 12op/widened trace pair.
# Usage: run_sampled.sh <NAME (traces/<NAME>_{12op,widened})> <waves>
set -euo pipefail
cd "$(dirname "$0")"
NAME=$1; W=$2; NCTA=$((W * 70))   # 70 SMs, 1 CTA/SM for this GEMM
for m in 12op widened; do
  ./run_sim.sh traces/${NAME}_$m ${NAME}_${m}_w$W -gpgpu_kernel_launch_latency 0 -gpgpu_max_cta $NCTA > /dev/null 2>&1 &
done
wait
a=$(grep -m1 '^gpu_tot_sim_cycle' sim_runs/${NAME}_12op_w$W/sim.log | awk '{print $3}')
b=$(grep -m1 '^gpu_tot_sim_cycle' sim_runs/${NAME}_widened_w$W/sim.log | awk '{print $3}')
python3 -c "print('RESULT $NAME waves=$W ctas=$NCTA 12op=%d widened=%d pct=%+.3f%%' % ($a, $b, 100.0*($b-$a)/$a))"
