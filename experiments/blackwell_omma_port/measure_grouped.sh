#!/usr/bin/env bash
# Grouped (MoE expert) GEMM: baseline vs correction-data TMA load (swc0) vs load + packing (swc3),
# real Qwen / DeepSeek expert configurations (same as nvfp4_microbench/code/grouped/run_grouped.sh):
#   Qwen     : 128 experts, per-expert M = tokens/16, gate/up N=768 K=2048, down N=2048 K=768
#   DeepSeek :  64 experts, per-expert M = tokens*6/64 (128-multiple), gate/up N=1408 K=2048, down N=2048 K=1408
# REPS interleaved reps (binary order rotated each rep), 50 iterations each; shapes split over 4 GPUs,
# all three binaries of a shape on the same GPU. Output: logs_grouped_raw.txt.<gpu> -> logs_grouped_raw.txt
set -uo pipefail
EXP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); cd "$EXP"
REPS=${REPS:-10}; PAD=${PAD:-0}
SHAPES=(
 "qwen_gateup_M2048 128 768 2048 128"  "qwen_down_M2048 128 2048 768 128"
 "qwen_gateup_M4096 256 768 2048 128"  "qwen_down_M4096 256 2048 768 128"
 "qwen_gateup_M8192 512 768 2048 128"  "qwen_down_M8192 512 2048 768 128"
 "ds_routed_gu_M2048 128 1408 2048 64" "ds_routed_down_M2048 128 2048 1408 64"
 "ds_routed_gu_M4096 384 1408 2048 64" "ds_routed_down_M4096 384 2048 1408 64"
 "ds_routed_gu_M8192 768 1408 2048 64" "ds_routed_down_M8192 768 2048 1408 64"
)
run_gpu() { g=$1; out=logs_grouped_raw.txt.$g; : > $out
  docker run --rm --gpus "\"device=$g\"" -v "$EXP/bin":/w -w /w --entrypoint bash nvcr.io/nvidia/pytorch:25.12-py3 -c "
    B=(grp_base_p$PAD grp_swc0_p$PAD grp_swc3_p$PAD)
    for rep in \$(seq 1 $REPS); do
      $(for i in "${!SHAPES[@]}"; do [ $((i % 4)) = $g ] || continue; set -- ${SHAPES[$i]}
        echo "for j in 0 1 2; do b=\${B[\$(( (j + rep) % 3 ))]}; echo \"### rep\$rep $1 \$b\";
              ./\$b --m=$2 --n=$3 --k=$4 --groups=$5 --iterations=50 --no_verif 2>&1 | grep -E 'Stages|Avg runtime|Kernel-only'; done;"
      done)
    done" >> $out 2>&1
}
for g in 0 1 2 3; do run_gpu $g & done; wait
cat logs_grouped_raw.txt.[0-3] > logs_grouped_raw.txt; echo DONE >> logs_grouped_raw.txt
