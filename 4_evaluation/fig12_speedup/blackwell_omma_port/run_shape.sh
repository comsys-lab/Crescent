#!/usr/bin/env bash
# Trace one GEMM shape of the mode-3 + tie build, build the 12op / widened traces,
# simulate both (SM120_RTX_PRO_4000, kernel launch latency 0) and print the pct.
# Usage: run_shape.sh <linear|qkt> <M> <N> <K>
set -euo pipefail
cd "$(dirname "$0")"; source ./env.sh
CFG=$1; M=$2; N=$3; K=$4
NAME=m3tie_${CFG}_${M}x${N}x${K}
BIN=./bin/nvfp4_microbench_mode3_${CFG}_tie
SASS=sass/nvfp4_microbench_mode3_${CFG}_tie.sass
PP=$ACCELSIM/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing
KID=9   # first launch of the GEMM kernel in nvfp4_microbench (--warmup=0 --iters=1 --no-check)

if ls traces/$NAME/traces/kernel-$KID-*.trace.xz > /dev/null 2>&1; then
  echo "== reuse existing trace traces/$NAME"
else
  date; echo "== trace $NAME"
  DYNAMIC_KERNEL_RANGE="$KID-$KID@.*GemmUniversal.*" TOOL_VERBOSE=0 \
    ./trace.sh traces/$NAME $BIN --m=$M --n=$N --k=$K --iters=1 --warmup=0 --no-check > /dev/null
  grep -aE "cuda error|Assert" traces/$NAME/tracer_stdout.txt && exit 1
fi
date; echo "== build 12op / widened traces"
python3 -I omma_operand_widen.py traces/$NAME/traces traces/$NAME $KID $SASS
for m in 12op widened; do
  $PP traces/${NAME}_$m/traces > traces/${NAME}_$m/postproc.txt 2>&1 &
done
wait
du -sh traces/${NAME}_12op/traces/*.tracez
date; echo "== simulate"
for m in 12op widened; do
  ./run_sim.sh traces/${NAME}_$m ${NAME}_$m -gpgpu_kernel_launch_latency 0 > /dev/null 2>&1 &
done
wait
a=$(grep -m1 '^gpu_tot_sim_cycle' sim_runs/${NAME}_12op/sim.log | awk '{print $3}')
b=$(grep -m1 '^gpu_tot_sim_cycle' sim_runs/${NAME}_widened/sim.log | awk '{print $3}')
grep -E '^gpgpu_simulation_time' sim_runs/${NAME}_12op/sim.log
python3 -c "print('RESULT $NAME 12op=%d widened=%d pct=%+.3f%%' % ($a, $b, 100.0*($b-$a)/$a))"
date
