#!/usr/bin/env bash
# Operand term with SM-count-independent traces (static scheduler, one tile per CTA; bin/*_tie_static),
# on SM100_B200_CLASS and SM120_RTX_PRO_4000. Same 6 shapes / 20 launch-latency pairs as logs_noise_*.csv.
# Output: logs_noise_st_<cfg>_<tag>.csv/.txt
set -uo pipefail
cd "$(dirname "$0")"; source ./env.sh
PP=$ACCELSIM/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing
KID=9
SHAPES="qproj:linear:2048:4096:4096 kv4096:linear:4096:1024:4096 dsrgu768:linear:768:1408:2048 qkt2048:qkt:2048:2048:128 qkt4096:qkt:4096:4096:128 qkt8192:qkt:8192:8192:128"
for s in $SHAPES; do IFS=: read TAG C M N K <<< "$s"; NAME=st_${TAG}
  echo "$(date '+%F %T') trace $NAME"
  DYNAMIC_KERNEL_RANGE="$KID-$KID@.*GemmUniversal.*" TOOL_VERBOSE=0 CUDA_VISIBLE_DEVICES=0 \
    ./trace.sh traces/$NAME ./bin/nvfp4_microbench_mode3_${C}_tie_static --m=$M --n=$N --k=$K --iters=1 --warmup=0 --no-check > /dev/null
  python3 -I omma_operand_widen.py traces/$NAME/traces traces/$NAME $KID sass/nvfp4_microbench_mode3_${C}_tie_static.sass > traces/$NAME/widen.txt 2>&1
  for m in 12op widened; do $PP traces/${NAME}_$m/traces > traces/${NAME}_$m/postproc.txt 2>&1; done
  rm -f traces/$NAME/traces/*.trace.xz
done
echo "$(date '+%F %T') simulate"
export MAX_SIMS=${MAX_SIMS:-56}
for cfg in SM100_B200_CLASS SM120_RTX_PRO_4000; do
  short=$([ $cfg = SM100_B200_CLASS ] && echo b200 || echo sm120)
  for s in $SHAPES; do TAG=${s%%:*}
    CFG=$cfg ./run_noise_pairs.sh traces/st_${TAG} st_${short}_${TAG} > logs_noise_st_${short}_${TAG}.txt 2>&1 &
    sleep 2
  done
done
wait; echo "$(date '+%F %T') done"
