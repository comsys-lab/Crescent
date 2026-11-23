#!/usr/bin/env bash
# Real-GPU packing cost per shape: GEMM ms of the patched CUTLASS GEMM with metadata TMA
# load only (NVFP4_META_SW_CORRECT unset/0, bin/*_swc0) vs TMA load + packing (=3),
# same SMEM (81920 B) and Stage count. 5 interleaved reps per build, default
# --iters=50 --warmup=5, --no-check. Shapes from shapes57.txt, round-robin over GPUs 0-3.
# Output: logs_packing_raw.csv (label,cfg,M,N,K,gpu,rep,build,gemm_ms)
set -uo pipefail
cd "$(dirname "$0")"
OUT=logs_packing_raw.csv
echo "label,cfg,M,N,K,gpu,rep,build,gemm_ms" > $OUT
run_gpu() {
  local gpu=$1 i=0
  while read -r LABEL CFG M N K; do
    i=$((i + 1)); [ $(( (i - 1) % 4 )) -eq $gpu ] || continue
    for rep in 1 2 3 4 5; do
      for b in swc0 swc3; do
        bin=./bin/nvfp4_microbench_mode3_${CFG}; [ $b = swc0 ] && bin=${bin}_swc0
        ms=$(CUDA_VISIBLE_DEVICES=$gpu $bin --m=$M --n=$N --k=$K --no-check 2>&1 | sed -nE 's/^ *nvfp4 GEMM +: +([0-9.]+) ms.*/\1/p')
        echo "$LABEL,$CFG,$M,$N,$K,$gpu,$rep,$b,$ms" >> $OUT.gpu$gpu
      done
    done
  done < shapes57.txt
}
for g in 0 1 2 3; do run_gpu $g & done
wait
cat $OUT.gpu0 $OUT.gpu1 $OUT.gpu2 $OUT.gpu3 >> $OUT
echo "$(date '+%F %T') done"
