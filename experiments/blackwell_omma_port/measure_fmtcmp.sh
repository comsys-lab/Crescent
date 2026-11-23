#!/usr/bin/env bash
# NVFP4 / MXFP4 / MXFP6 / NVFP6(emulated) / Czardas(MXFP4) timing for the format comparison.
# Dense (3 interleaved reps): Linear shapes (non-MoE, shapes57) and QK^T shapes; PV (M x 128 x M) NVFP4 / MXFP4.
#   natural Stage count : fc_nv4_p0 fc_mx4_p0 fc_mx6{lin,qkt}_p0 fc_nv6{lin,qkt}_p0
#   Stage-matched (4)   : fc_mx4_p16384 fc_czmx{lin,qkt}{0,3}_p16384
# Grouped (3 reps): grp_base_p0 grp_mx1_p0 grp_mx2_p0 grp_nv6emu_p0 | grp_mx1_p8192 grp_czmx_swc{0,3}_p8192
# Quantize (3 reps): quant_fp6_bench at every activation tensor (Linear inputs, Q/K per head, P/V per head,
#   MoE full-batch inputs).
set -uo pipefail
EXP=$(cd "$(dirname "$0")" && pwd); cd "$EXP"
echo "$(date '+%F %T') dense"
OUT=logs_fc_dense.txt; rm -f $OUT $OUT.*
{ grep -vE '^(qwen_gateup|qwen_down|ds_routed_gu|ds_routed_down)_' shapes57.txt | awk '{print $1, $2, $3, $4, $5}'
  for M in 2048 4096 8192; do echo "pv_M$M pv $M 128 $M"; done; } > fc_shapes.txt
d() { local g=$1 i=0
  while read -r LABEL CFG M N K; do i=$((i + 1)); [ $(( (i - 1) % 4 )) -eq $g ] || continue
    case $CFG in
      linear) B="fc_nv4_p0 fc_mx4_p0 fc_mx6lin_p0 fc_nv6lin_p0 fc_mx4_p16384 fc_czmxlin0_p16384 fc_czmxlin3_p16384" ;;
      qkt)    B="fc_nv4_p0 fc_mx4_p0 fc_mx6qkt_p0 fc_nv6qkt_p0 fc_mx4_p16384 fc_czmxqkt0_p16384 fc_czmxqkt3_p16384" ;;
      pv)     B="fc_nv4_p0 fc_mx4_p0" ;;
    esac
    for rep in 1 2 3; do for b in $B; do
      echo "### rep$rep $LABEL $b $(CUDA_VISIBLE_DEVICES=$g ./bin/$b --m=$M --n=$N --k=$K --iters=30 --warmup=5 2>&1 | grep -o 'gemm_ms=[0-9.]*\|rror.*' | head -1)" >> $OUT.$g
    done; done
  done < fc_shapes.txt; }
for g in 0 1 2 3; do d $g & done; wait; cat $OUT.[0-3] > $OUT; echo DONE >> $OUT

echo "$(date '+%F %T') quant"
OUT=logs_fc_quant.txt; rm -f $OUT $OUT.*
{ awk '$2=="linear" {print $1"_a", $3, $5}' fc_shapes.txt
  for M in 2048 4096 8192; do echo "qk_M$M $M 128"; echo "p_M$M $M $M"; echo "v_M$M 128 $M"; done
  awk '{print $1, $2, $4}' moe_quant_shapes.txt; } > fc_quant_shapes.txt
q() { local g=$1 i=0
  while read -r LABEL M K; do i=$((i + 1)); [ $(( (i - 1) % 4 )) -eq $g ] || continue
    for rep in 1 2 3; do echo "### rep$rep $LABEL $(CUDA_VISIBLE_DEVICES=$g ./bin/quant_fp6_bench --m=$M --k=$K 2>&1 | tail -1)" >> $OUT.$g; done
  done < fc_quant_shapes.txt; }
for g in 0 1 2 3; do q $g & done; wait; cat $OUT.[0-3] > $OUT; echo DONE >> $OUT

echo "$(date '+%F %T') grouped"
SHAPES=(
 "qwen_gateup_M2048 128 768 2048 128"  "qwen_down_M2048 128 2048 768 128"
 "qwen_gateup_M4096 256 768 2048 128"  "qwen_down_M4096 256 2048 768 128"
 "qwen_gateup_M8192 512 768 2048 128"  "qwen_down_M8192 512 2048 768 128"
 "ds_routed_gu_M2048 128 1408 2048 64" "ds_routed_down_M2048 128 2048 1408 64"
 "ds_routed_gu_M4096 384 1408 2048 64" "ds_routed_down_M4096 384 2048 1408 64"
 "ds_routed_gu_M8192 768 1408 2048 64" "ds_routed_down_M8192 768 2048 1408 64"
)
OUT=logs_fc_grouped.txt; rm -f $OUT $OUT.*
gr() { local g=$1
  docker run --rm --gpus "\"device=$g\"" -v "$EXP/bin":/w -w /w --entrypoint bash nvcr.io/nvidia/pytorch:25.12-py3 -c "
    B=(grp_base_p0 grp_mx1_p0 grp_mx2_p0 grp_nv6emu_p0 grp_mx1_p8192 grp_czmx_swc0_p8192 grp_czmx_swc3_p8192)
    for rep in 1 2 3; do
      $(for i in "${!SHAPES[@]}"; do [ $((i % 4)) = $g ] || continue; set -- ${SHAPES[$i]}
        echo "for j in 0 1 2 3 4 5 6; do b=\${B[\$(( (j + rep) % 7 ))]}; echo \"### rep\$rep $1 \$b\";
              ./\$b --m=$2 --n=$3 --k=$4 --groups=$5 --iterations=50 --no_verif 2>&1 | grep -E 'Stages|Avg runtime|Kernel-only'; done;"
      done)
    done" >> $OUT.$g 2>&1; }
for g in 0 1 2 3; do gr $g & done; wait; cat $OUT.[0-3] > $OUT; echo DONE >> $OUT
echo "$(date '+%F %T') done"
