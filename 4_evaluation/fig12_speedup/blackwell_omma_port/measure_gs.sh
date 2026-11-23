#!/usr/bin/env bash
# New group sizes (Linear 1-of-8, QK^T 1-of-16): all 57 shapes, interleaved reps of
#   baseline  : unpatched CUTLASS, NVFP4_META_ENABLE=0, default pad (bin/nvfp4_microbench_baseline_padded)
#   sel       : patched CUTLASS, metadata TMA load only (bin/gs_{lin8,qkt16}_swc0)
#   pack      : patched CUTLASS, TMA load + packing      (bin/gs_{lin8,qkt16}_swc3)
# 5 reps (10 for MoE routed-expert shapes). Output blocks
#   "### rep<r> <label> <build> n= k= m=" + the binary's GEMM / quantize lines.
# Usage: measure_gs.sh "<gpu list>"
set -uo pipefail
cd "$(dirname "$0")"
GPUS=($1); OUT=logs_gs_raw.txt; : > $OUT
run_gpu() {
  local slot=$1 gpu=${GPUS[$1]} i=0
  while read -r LABEL CFG M N K; do
    i=$((i + 1)); [ $(( (i - 1) % ${#GPUS[@]} )) -eq $slot ] || continue
    sel=gs_lin8; [ $CFG = qkt ] && sel=gs_qkt16
    reps=5; [[ $LABEL =~ ^(qwen_gateup|qwen_down|ds_routed_gu|ds_routed_down)_ ]] && reps=10
    for rep in $(seq 1 $reps); do
      for b in baseline sel pack; do
        case $b in baseline) bin=nvfp4_microbench_baseline_padded ;; sel) bin=${sel}_swc0 ;; pack) bin=${sel}_swc3 ;; esac
        { echo "### rep$rep $LABEL $b n=$N k=$K m=$M"
          CUDA_VISIBLE_DEVICES=$gpu ./bin/$bin --m=$M --n=$N --k=$K --no-check 2>&1 | grep -E "nvfp4 GEMM +:|DRAM store\+reload|with store \(DRAM\)"; } >> $OUT.$slot
      done
    done
  done < shapes57.txt
}
for s in $(seq 0 $((${#GPUS[@]} - 1))); do run_gpu $s & done
wait
cat $OUT.* >> $OUT
echo "$(date '+%F %T') done"
