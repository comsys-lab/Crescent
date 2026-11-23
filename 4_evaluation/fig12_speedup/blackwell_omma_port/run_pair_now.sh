#!/usr/bin/env bash
# Simulate an already converted all_<label>_{12op,widened} pair and append its RESULT line
# (same format as run_all_shapes.sh). Usage: run_pair_now.sh <label> <M> <N> <K>
set -uo pipefail
cd "$(dirname "$0")"
LABEL=$1; M=$2; N=$3; K=$4; NAME=all_$LABEL; T=traces/$NAME
for m in 12op widened; do
  ./run_sim.sh "${T}_$m" ${NAME}_$m -gpgpu_kernel_launch_latency 0 > /dev/null 2>&1 &
done
wait
a=$(grep -m1 '^gpu_tot_sim_cycle' sim_runs/${NAME}_12op/sim.log | awk '{print $3}')
b=$(grep -m1 '^gpu_tot_sim_cycle' sim_runs/${NAME}_widened/sim.log | awk '{print $3}')
if [ -n "$a" ] && [ -n "$b" ]; then
  python3 -c "print('RESULT $NAME label=$LABEL M=$M N=$N K=$K 12op=%d widened=%d pct=%+.4f%%' % ($a, $b, 100.0*($b-$a)/$a))" >> logs_all_results.txt
else
  echo "FAILED $NAME" >> logs_all_results.txt
fi
