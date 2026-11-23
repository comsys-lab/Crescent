#!/bin/bash
cd "$(dirname "$0")"
SHAPES="qproj:linear:2048:4096:4096 kv4096:linear:4096:1024:4096 dsrgu768:linear:768:1408:2048 qkt2048:qkt:2048:2048:128 qkt4096:qkt:4096:4096:128 qkt8192:qkt:8192:8192:128"
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
