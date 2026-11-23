#!/usr/bin/env bash
# Repeat the MoE routed-expert microbench (the shapes whose single-shot deltas sit at the
# noise floor) exactly as in execution_time: baseline = unpatched CUTLASS,
# NVFP4_META_ENABLE=0, default NVFP4_SMEM_PAD (nvfp4_microbench_baseline); selection =
# metaA-patched CUTLASS, NVFP4_SMEM_PAD default (nvfp4_microbench_linear). 10 interleaved
# reps. Output blocks "### rep<r> <label> <binary> n= k= m=" + the binary's stdout.
# Usage: measure_moe_repeat.sh <gpu list, e.g. "0 2">
set -uo pipefail
cd "$(dirname "$0")"
GPUS=($1); OUT=logs_moe_repeat_raw.txt; : > $OUT
grep -E "^(qwen_gateup|qwen_down|ds_routed_gu|ds_routed_down)_" shapes57.txt > moe12.txt
run_gpu() {
  local slot=$1 gpu=${GPUS[$1]} i=0
  while read -r LABEL CFG M N K; do
    i=$((i + 1)); [ $(( (i - 1) % ${#GPUS[@]} )) -eq $slot ] || continue
    for rep in $(seq 1 10); do
      for b in baseline linear; do
        bin=./bin/nvfp4_microbench_baseline_padded; [ $b = linear ] && bin=./bin/nvfp4_microbench_mode3_linear_swc0
        { echo "### rep$rep $LABEL nvfp4_microbench_$b n=$N k=$K m=$M"
          CUDA_VISIBLE_DEVICES=$gpu $bin --m=$M --n=$N --k=$K --no-check 2>&1 | grep -E "nvfp4 GEMM +:|DRAM store\+reload|with store \(DRAM\)"; } >> $OUT.$slot
      done
    done
  done < moe12.txt
}
for s in $(seq 0 $((${#GPUS[@]} - 1))); do run_gpu $s & done
wait
cat $OUT.* >> $OUT
echo "$(date '+%F %T') done"
