#!/usr/bin/env bash
# Final pre-packed (NVFP4_META_PREPACK) measurement for the Linear terms; QK^T keeps gs_qkt16 (logs_gs_raw.txt).
# Phase 1 dense Linear, 36 non-MoE shapes, 3 interleaved reps: baseline_padded / gs_lin8pp_swc0 / gs_lin8pp_swc3 -> logs_pp_raw.txt
# Phase 2 MoE full-batch quantize side, PREPACK store, gs_lin8pp_swc0, 3 reps                            -> logs_moe_quant_pp_raw.txt
# Phase 3 grouped GEMM, 12 MoE shapes, 5 interleaved reps: grp_base / grp_pp_swc0 / grp_pp_swc3            -> logs_grouped_pp_raw.txt
set -uo pipefail
EXP=$(cd "$(dirname "$0")" && pwd); cd "$EXP"
echo "$(date '+%F %T') phase 1"
OUT=logs_pp_raw.txt; rm -f $OUT $OUT.*
p1() { local g=$1 i=0
  grep -vE '^(qwen_gateup|qwen_down|ds_routed_gu|ds_routed_down)_' shapes57.txt | awk '$2=="linear"' | while read -r LABEL CFG M N K; do
    i=$((i + 1)); [ $(( (i - 1) % 4 )) -eq $g ] || continue
    for rep in 1 2 3; do for j in 0 1 2; do
      case $(( (j + rep) % 3 )) in 0) b=baseline; bin=nvfp4_microbench_baseline_padded ;; 1) b=sel; bin=gs_lin8pp_swc0 ;; 2) b=pack; bin=gs_lin8pp_swc3 ;; esac
      { echo "### rep$rep $LABEL $b n=$N k=$K m=$M"
        CUDA_VISIBLE_DEVICES=$g ./bin/$bin --m=$M --n=$N --k=$K --no-check 2>&1 | grep -E "nvfp4 GEMM +:|DRAM store\+reload|with store \(DRAM\)"; } >> $OUT.$g
    done; done
  done; }
for g in 0 1 2 3; do p1 $g & done; wait; cat $OUT.[0-3] > $OUT; echo DONE >> $OUT
echo "$(date '+%F %T') phase 2"
OUT=logs_moe_quant_pp_raw.txt; rm -f $OUT $OUT.*
p2() { local g=$1 i=0
  while read -r LABEL M N K; do i=$((i + 1)); [ $(( (i - 1) % 4 )) -eq $g ] || continue
    for rep in 1 2 3; do
      { echo "### rep$rep $LABEL sel n=$N k=$K m=$M"
        CUDA_VISIBLE_DEVICES=$g ./bin/gs_lin8pp_swc0 --m=$M --n=$N --k=$K --no-check 2>&1 | grep -E "nvfp4 GEMM +:|DRAM store\+reload|with store \(DRAM\)"; } >> $OUT.$g
    done
  done < moe_quant_shapes.txt; }
for g in 0 1 2 3; do p2 $g & done; wait; cat $OUT.[0-3] > $OUT; echo DONE >> $OUT
echo "$(date '+%F %T') phase 3"
SHAPES=(
 "qwen_gateup_M2048 128 768 2048 128"  "qwen_down_M2048 128 2048 768 128"
 "qwen_gateup_M4096 256 768 2048 128"  "qwen_down_M4096 256 2048 768 128"
 "qwen_gateup_M8192 512 768 2048 128"  "qwen_down_M8192 512 2048 768 128"
 "ds_routed_gu_M2048 128 1408 2048 64" "ds_routed_down_M2048 128 2048 1408 64"
 "ds_routed_gu_M4096 384 1408 2048 64" "ds_routed_down_M4096 384 2048 1408 64"
 "ds_routed_gu_M8192 768 1408 2048 64" "ds_routed_down_M8192 768 2048 1408 64"
)
OUT=logs_grouped_pp_raw.txt; rm -f $OUT $OUT.*
p3() { local g=$1
  docker run --rm --gpus "\"device=$g\"" -v "$EXP/bin":/w -w /w --entrypoint bash nvcr.io/nvidia/pytorch:25.12-py3 -c "
    B=(grp_base_p0 grp_pp_swc0_p0 grp_pp_swc3_p0)
    for rep in 1 2 3 4 5; do
      $(for i in "${!SHAPES[@]}"; do [ $((i % 4)) = $g ] || continue; set -- ${SHAPES[$i]}
        echo "for j in 0 1 2; do b=\${B[\$(( (j + rep) % 3 ))]}; echo \"### rep\$rep $1 \$b\";
              ./\$b --m=$2 --n=$3 --k=$4 --groups=$5 --iterations=50 --no_verif 2>&1 | grep -E 'Stages|Avg runtime|Kernel-only'; done;"
      done)
    done" >> $OUT.$g 2>&1; }
for g in 0 1 2 3; do p3 $g & done; wait; cat $OUT.[0-3] > $OUT; echo DONE >> $OUT
echo "$(date '+%F %T') done"
