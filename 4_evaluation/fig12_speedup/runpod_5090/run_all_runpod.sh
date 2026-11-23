#!/usr/bin/env bash
# Re-measure every real-GPU input of the execution-time tables on one RTX 5090 (RunPod, no docker).
# Pod image: nvcr.io/nvidia/pytorch:25.12-py3 (same nvcc / cuBLAS 13 / torch as the original runs).
# Same binaries, command lines, rep counts and output formats as the RTX PRO 4000 runs
# (4_evaluation/fig12_speedup/blackwell_omma_port/measure_{gs,pp_final,fmtcmp}.sh, measure_moe_other.py,
# py/{ab_truebase_vs_sel_repeat,bf16_attn_repeat}.py), sequential on one GPU.
# Not re-measured: Accel-Sim operand fraction (option (a): reused from the sm_120 simulation).
#
# Shapes are split over all visible GPUs (shape i -> GPU i mod NG, all builds of a shape on the same GPU),
# as the original runs did over 4 GPUs. GPUS="0 1" to choose.
# Usage:  export HF_TOKEN=hf_...      (Llama-3.1-8B is gated)
#         nohup setsid ./run_all_runpod.sh > run.log 2>&1 &
# Resumable: a finished phase leaves results/<file> ending in DONE and is skipped on rerun.
# Output: results/ (copy the whole directory back).
set -uo pipefail
B=$(cd "$(dirname "$0")" && pwd); cd "$B"
GPUS=(${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr "\n" " ")}); NG=${#GPUS[@]}
export CUDA_VISIBLE_DEVICES=${GPUS[0]}   # sanity / single-GPU parts
R=$B/results; mkdir -p $R/log
BIN=$B/bin; export BUNDLE_BIN=$BIN
PY=${PY:-python3}   # e.g. PY=/workspace/venv/bin/python (torch 2.10 cu130)
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
done_() { [ -f "$1" ] && tail -1 "$1" | grep -q '^DONE$'; }
MB="nvfp4 GEMM +:|DRAM store\+reload|with store \(DRAM\)"
NONMOE='^(qwen_gateup|qwen_down|ds_routed_gu|ds_routed_down)_'
GSHAPES=(
 "qwen_gateup_M2048 128 768 2048 128"  "qwen_down_M2048 128 2048 768 128"
 "qwen_gateup_M4096 256 768 2048 128"  "qwen_down_M4096 256 2048 768 128"
 "qwen_gateup_M8192 512 768 2048 128"  "qwen_down_M8192 512 2048 768 128"
 "ds_routed_gu_M2048 128 1408 2048 64" "ds_routed_down_M2048 128 2048 1408 64"
 "ds_routed_gu_M4096 384 1408 2048 64" "ds_routed_down_M4096 384 2048 1408 64"
 "ds_routed_gu_M8192 768 1408 2048 64" "ds_routed_down_M8192 768 2048 1408 64"
)

# run_slots <out> <fn>: fn <slot> writes <out>.<slot>; slots run in parallel, one per GPU, then merged.
run_slots() { local out=$1 fn=$2; rm -f $out $out.*
  local pids=() sl; for ((sl=0; sl<NG; sl++)); do ( export CUDA_VISIBLE_DEVICES=${GPUS[$sl]}; $fn $sl > $out.$sl ) & pids+=($!); done
  wait "${pids[@]}"
  cat $out.* > $out; rm -f $out.*; echo DONE >> $out; }
mine() { [ $(( ($1) % NG )) -eq $2 ]; }   # mine <shape index> <slot>

# ---------------------------------------------------------------- 0. environment + sanity
echo "$(ts) phase 0: environment"
chmod +x $BIN/* 2>/dev/null
nvidia-smi > $R/nvidia_smi.txt; nvidia-smi -q >> $R/nvidia_smi.txt
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.max.sm,clocks.max.mem --format=csv | tee $R/gpu.txt
grep -q "5090" $R/gpu.txt || { echo "WARNING: GPU is not an RTX 5090"; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "ERROR: other processes are using the GPU"; nvidia-smi; exit 1; fi
$BIN/devprops > $R/devprops.txt 2>&1; cat $R/devprops.txt
fail=0
for b in nvfp4_microbench_baseline_padded gs_qkt16_swc0 gs_qkt16_swc3 gs_lin8pp_swc0 gs_lin8pp_swc3 nvfp4_microbench_qkt nvfp4_microbench_baseline; do
  $BIN/$b --m=512 --n=512 --k=1024 --iters=2 --warmup=1 --no-check 2>&1 | grep "nvfp4 GEMM\|bf16 GEMM" > /dev/null || { echo "SANITY FAIL $b"; fail=1; }; done
for b in fc_nv4_p0 fc_mx4_p0 fc_mx6lin_p0 fc_nv6lin_p0 fc_mx6qkt_p0 fc_nv6qkt_p0 fc_mx4_p16384 fc_czmxlin0_p16384 fc_czmxlin3_p16384 fc_czmxqkt0_p16384 fc_czmxqkt3_p16384; do
  $BIN/$b --m=512 --n=512 --k=1024 --iters=2 --warmup=1 2>&1 | grep "gemm_ms=" > /dev/null || { echo "SANITY FAIL $b"; fail=1; }; done
for b in grp_base_p0 grp_pp_swc0_p0 grp_pp_swc3_p0 grp_mx1_p0 grp_mx2_p0 grp_nv6emu_p0 grp_mx1_p8192 grp_czmx_swc0_p8192 grp_czmx_swc3_p8192; do
  $BIN/$b --m=128 --n=768 --k=2048 --groups=8 --iterations=2 --no_verif 2>&1 | grep "Avg runtime" > /dev/null || { echo "SANITY FAIL $b"; fail=1; }; done
$BIN/quant_fp6_bench --m=2048 --k=4096 2>&1 | tail -1 | grep -q RESULT || { echo "SANITY FAIL quant_fp6_bench"; fail=1; }
[ $fail = 0 ] || { echo "$(ts) sanity failed, stopping"; exit 1; }
echo "$(ts) sanity ok"

# Python deps (as in the original runs) + Llama download in the background while the GEMM phases run
[ -n "${SKIP_PIP:-}" ] || pip install -q --no-deps "tokenizers==0.22.1" "transformers<5" "huggingface_hub<1.0" accelerate 2>&1 | grep -i error
$PY -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)" | tee $R/py_env.txt
LLAMA_MODEL=${LLAMA_MODEL:-meta-llama/Llama-3.1-8B}; export LLAMA_MODEL
rm -f $R/log/llama_download.ok
# LLAMA_COPY_MARKER=<file>: weights are being copied into the HF cache from elsewhere; wait for the marker, then
# resolve them offline (no HF token on the pod).
( if [ -n "${LLAMA_COPY_MARKER:-}" ]; then while [ ! -f "$LLAMA_COPY_MARKER" ]; do sleep 20; done; export HF_HUB_OFFLINE=1; fi
  $PY -c "from huggingface_hub import snapshot_download as s; print(s('$LLAMA_MODEL', allow_patterns=['*.safetensors','*.json']))" \
  > $R/log/llama_download.log 2>&1 && touch $R/log/llama_download.ok ) &
DL=$!

# ---------------------------------------------------------------- 1. gs8_16_raw.txt
# QK^T 1-of-16 (baseline_padded / gs_qkt16_swc0 / swc3) and the dense-Linear NVFP4 baseline (aq + GEMM,
# denominator), 5 reps, as measure_gs.sh. (Linear sel/pack of that script are superseded by phase 2,
# MoE routed shapes by the grouped GEMM -> not run.)
p1() { local sl=$1 i=0
  grep -vE "$NONMOE" shapes57.txt | while read -r LABEL CFG M N K; do i=$((i + 1)); mine $((i - 1)) $sl || continue
    BS=baseline; [ $CFG = qkt ] && BS="baseline sel pack"
    for rep in 1 2 3 4 5; do for b in $BS; do
      case $b in baseline) bin=nvfp4_microbench_baseline_padded ;; sel) bin=gs_qkt16_swc0 ;; pack) bin=gs_qkt16_swc3 ;; esac
      echo "### rep$rep $LABEL $b n=$N k=$K m=$M"
      $BIN/$bin --m=$M --n=$N --k=$K --no-check 2>&1 | grep -E "$MB"
    done; done
  done; }
OUT=$R/gs8_16_raw.txt; done_ $OUT || { echo "$(ts) phase 1: gs8_16_raw"; run_slots $OUT p1; }

# ---------------------------------------------------------------- 2. pre-packed Linear, MoE quantize, grouped (measure_pp_final.sh)
p2a() { local sl=$1 i=0
  grep -vE "$NONMOE" shapes57.txt | awk '$2=="linear"' | while read -r LABEL CFG M N K; do i=$((i + 1)); mine $((i - 1)) $sl || continue
    for rep in 1 2 3; do for j in 0 1 2; do
      case $(( (j + rep) % 3 )) in 0) b=baseline; bin=nvfp4_microbench_baseline_padded ;; 1) b=sel; bin=gs_lin8pp_swc0 ;; 2) b=pack; bin=gs_lin8pp_swc3 ;; esac
      echo "### rep$rep $LABEL $b n=$N k=$K m=$M"
      $BIN/$bin --m=$M --n=$N --k=$K --no-check 2>&1 | grep -E "$MB"
    done; done
  done; }
OUT=$R/pp_raw.txt; done_ $OUT || { echo "$(ts) phase 2a: pp_raw"; run_slots $OUT p2a; }

p2b() { local sl=$1 i=0
  while read -r LABEL M N K; do i=$((i + 1)); mine $((i - 1)) $sl || continue
    for rep in 1 2 3; do
      echo "### rep$rep $LABEL sel n=$N k=$K m=$M"
      $BIN/gs_lin8pp_swc0 --m=$M --n=$N --k=$K --no-check 2>&1 | grep -E "$MB"
    done
  done < moe_quant_shapes.txt; }
OUT=$R/moe_quant_pp_raw.txt; done_ $OUT || { echo "$(ts) phase 2b: moe_quant_pp_raw"; run_slots $OUT p2b; }

p2c() { local sl=$1 i GB=(grp_base_p0 grp_pp_swc0_p0 grp_pp_swc3_p0)
  for i in "${!GSHAPES[@]}"; do mine $i $sl || continue; set -- ${GSHAPES[$i]}
    for rep in 1 2 3 4 5; do for j in 0 1 2; do b=${GB[$(( (j + rep) % 3 ))]}
      echo "### rep$rep $1 $b"
      $BIN/$b --m=$2 --n=$3 --k=$4 --groups=$5 --iterations=50 --no_verif 2>&1 | grep -E 'Stages|Avg runtime|Kernel-only'
    done; done
  done; }
OUT=$R/grouped_pp_raw.txt; done_ $OUT || { echo "$(ts) phase 2c: grouped_pp_raw"; run_slots $OUT p2c; }

# ---------------------------------------------------------------- 3. format comparison (measure_fmtcmp.sh)
{ grep -vE "$NONMOE" shapes57.txt | awk '{print $1, $2, $3, $4, $5}'
  for M in 2048 4096 8192; do echo "pv_M$M pv $M 128 $M"; done; } > fc_shapes.txt
p3a() { local sl=$1 i=0
  while read -r LABEL CFG M N K; do i=$((i + 1)); mine $((i - 1)) $sl || continue
    case $CFG in
      linear) BS="fc_nv4_p0 fc_mx4_p0 fc_mx6lin_p0 fc_nv6lin_p0 fc_mx4_p16384 fc_czmxlin0_p16384 fc_czmxlin3_p16384" ;;
      qkt)    BS="fc_nv4_p0 fc_mx4_p0 fc_mx6qkt_p0 fc_nv6qkt_p0 fc_mx4_p16384 fc_czmxqkt0_p16384 fc_czmxqkt3_p16384" ;;
      pv)     BS="fc_nv4_p0 fc_mx4_p0" ;;
    esac
    for rep in 1 2 3; do for b in $BS; do
      echo "### rep$rep $LABEL $b $($BIN/$b --m=$M --n=$N --k=$K --iters=30 --warmup=5 2>&1 | grep -o 'gemm_ms=[0-9.]*\|rror.*' | head -1)"
    done; done
  done < fc_shapes.txt; }
OUT=$R/fc_dense.txt; done_ $OUT || { echo "$(ts) phase 3a: fc_dense"; run_slots $OUT p3a; }

{ awk '$2=="linear" {print $1"_a", $3, $5}' fc_shapes.txt
  for M in 2048 4096 8192; do echo "qk_M$M $M 128"; echo "p_M$M $M $M"; echo "v_M$M 128 $M"; done
  awk '{print $1, $2, $4}' moe_quant_shapes.txt; } > fc_quant_shapes.txt
p3b() { local sl=$1 i=0
  while read -r LABEL M K; do i=$((i + 1)); mine $((i - 1)) $sl || continue
    for rep in 1 2 3; do echo "### rep$rep $LABEL $($BIN/quant_fp6_bench --m=$M --k=$K 2>&1 | tail -1)"; done
  done < fc_quant_shapes.txt; }
OUT=$R/fc_quant.txt; done_ $OUT || { echo "$(ts) phase 3b: fc_quant"; run_slots $OUT p3b; }

p3c() { local sl=$1 i FB=(grp_base_p0 grp_mx1_p0 grp_mx2_p0 grp_nv6emu_p0 grp_mx1_p8192 grp_czmx_swc0_p8192 grp_czmx_swc3_p8192)
  for i in "${!GSHAPES[@]}"; do mine $i $sl || continue; set -- ${GSHAPES[$i]}
    for rep in 1 2 3; do for j in 0 1 2 3 4 5 6; do b=${FB[$(( (j + rep) % 7 ))]}
      echo "### rep$rep $1 $b"
      $BIN/$b --m=$2 --n=$3 --k=$4 --groups=$5 --iterations=50 --no_verif 2>&1 | grep -E 'Stages|Avg runtime|Kernel-only'
    done; done
  done; }
OUT=$R/fc_grouped.txt; done_ $OUT || { echo "$(ts) phase 3c: fc_grouped"; run_slots $OUT p3c; }

# ---------------------------------------------------------------- 4. Llama attention split (qkt_pv_repeat / bf16_qkt_pv_repeat)
# NVFP4 QK^T / PV true baseline: the original nvfp4_microbench_truebase binary no longer exists; fc_nv4_p0 is
# the same thing (pristine CUTLASS NVFP4 GEMM, no padding, natural Stage count), written in the parser's format.
# Small (seconds per run) -> one GPU; the BF16 cuBLAS split runs on the next GPU at the same time.
p4a() {
  for rep in 1 2 3 4 5; do for m in 2048 4096 8192; do
    echo "### rep$rep QKT_true_base m=$m"
    echo "  nvfp4 GEMM       :   $($BIN/fc_nv4_p0 --m=$m --n=$m --k=128 --iters=100 --warmup=10 2>&1 | grep -o 'gemm_ms=[0-9.]*' | cut -d= -f2) ms/iter   (fc_nv4_p0)"
    echo "### rep$rep QKT_sel m=$m"
    $BIN/nvfp4_microbench_qkt --m=$m --n=$m --k=128 --iters=100 --warmup=10 --no-check | grep -E "nvfp4 GEMM |DRAM store\+reload"
    echo "### rep$rep PV_true_base m=$m"
    echo "  nvfp4 GEMM       :   $($BIN/fc_nv4_p0 --m=$m --n=128 --k=$m --iters=100 --warmup=10 2>&1 | grep -o 'gemm_ms=[0-9.]*' | cut -d= -f2) ms/iter   (fc_nv4_p0)"
  done; done; echo DONE; }
p4b() {
  for rep in 1 2 3 4 5; do for m in 2048 4096 8192; do
    echo "### rep$rep QKT_bf16 m=$m"
    $BIN/nvfp4_microbench_baseline --m=$m --n=$m --k=128 --iters=50 --warmup=10 --no-check | grep "bf16 GEMM"
    echo "### rep$rep PV_bf16 m=$m"
    $BIN/nvfp4_microbench_baseline --m=$m --n=128 --k=$m --iters=50 --warmup=10 --no-check | grep "bf16 GEMM"
  done; done; echo DONE; }
G2=${GPUS[$(( 1 % NG ))]}
P4=()
done_ $R/qkt_pv_repeat.txt || { echo "$(ts) phase 4a: qkt_pv_repeat"; ( CUDA_VISIBLE_DEVICES=${GPUS[0]} p4a > $R/qkt_pv_repeat.txt ) & P4+=($!); }
done_ $R/bf16_qkt_pv_repeat.txt || { echo "$(ts) phase 4b: bf16_qkt_pv_repeat"; ( CUDA_VISIBLE_DEVICES=$G2 p4b > $R/bf16_qkt_pv_repeat.txt ) & P4+=($!); }
[ ${#P4[@]} -gt 0 ] && wait "${P4[@]}"

# ---------------------------------------------------------------- 5+6. Qwen/DeepSeek non-GEMM ops (GPU 2) || Llama full model (GPU 1)
OUT=$R/moe_other.txt; P5=
if ! done_ $OUT; then echo "$(ts) phase 5: moe_other (GPU $G2)"
  ( CUDA_VISIBLE_DEVICES=$G2 $PY py/measure_moe_other.py > $OUT 2> $R/log/moe_other.err && echo DONE >> $OUT ) &
  P5=$!; [ $NG -ge 2 ] || wait $P5; fi

echo "$(ts) phase 6: waiting for Llama download"; while kill -0 $DL 2>/dev/null; do sleep 10; done
[ -f $R/log/llama_download.ok ] || { echo "Llama download failed, see results/log/llama_download.log"; [ -n "$P5" ] && wait $P5; exit 1; }
export PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPUS[0]}
[ -n "${LLAMA_COPY_MARKER:-}" ] && export HF_HUB_OFFLINE=1
FILT='Warning\|registered\|operator\|dispatch\|kernel\|self.m.impl\|Loading'
OUT=$R/log/linear_stage_corrected_5rep.log
if ! done_ $OUT; then echo "$(ts) phase 6a: Llama Linear true-baseline forward"
  $PY py/ab_truebase_vs_sel_repeat.py 2048 4096 8192 2>&1 | grep -v "$FILT" > $OUT; grep -q "^rep4 M= *8192" $OUT && echo DONE >> $OUT; fi
OUT=$R/log/bf16_real_attention_5rep.log
if ! done_ $OUT; then echo "$(ts) phase 6b: Llama BF16 attention"
  $PY py/bf16_attn_repeat.py 2048 4096 8192 2>&1 | grep -v "$FILT" > $OUT; grep -q "^rep4 M= *8192" $OUT && echo DONE >> $OUT; fi
[ -n "$P5" ] && wait $P5

nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,power.draw --format=csv >> $R/gpu.txt
for f in gs8_16_raw.txt pp_raw.txt moe_quant_pp_raw.txt grouped_pp_raw.txt fc_dense.txt fc_quant.txt fc_grouped.txt qkt_pv_repeat.txt bf16_qkt_pv_repeat.txt moe_other.txt log/linear_stage_corrected_5rep.log log/bf16_real_attention_5rep.log; do
  done_ $R/$f && echo "ok          $f" || echo "INCOMPLETE  $f"; done
echo "$(ts) all done"
