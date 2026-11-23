#!/usr/bin/env bash
# Grouped (MoE expert) GEMM, CUTLASS 79d driver (src/grouped_meta.cu) against cutlass_mode3 with the
# grouped correction-data patch (patches/patch_grouped_meta.py). NGC 25.12 (nvcc 13.1).
# Usage: build_grouped.sh PAD   -> bin/grp_base_p$PAD, bin/grp_swc0_p$PAD, bin/grp_swc3_p$PAD
#   base : NVFP4_META_GROUPED=0 (collective compiles as unpatched)
#   swc0 : correction-data TMA load only; swc3 : TMA load + packing (Linear 1-of-8)
set -euo pipefail
EXP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PAD=${1:-0}
docker run --rm --gpus all -v "$EXP/src":/src:ro -v "$EXP/cutlass_mode3":/cutlass:ro -v "$EXP/bin":/out -w /src \
  --entrypoint bash nvcr.io/nvidia/pytorch:25.12-py3 -c '
  set -e
  b() { n=$1; shift
    nvcc -std=c++17 -gencode=arch=compute_120a,code=sm_120a -O3 --expt-relaxed-constexpr --expt-extended-lambda \
         -DNVFP4_SMEM_PAD='$PAD' "$@" -I/cutlass/include -I/cutlass/tools/util/include -I/cutlass/examples/common \
         grouped_meta.cu -o /out/$n &
  }
  b grp_base_p'$PAD'
  b grp_swc0_p'$PAD' -DNVFP4_META_GROUPED=1 -DNVFP4_META_GROUP_SIZE=8 -DNVFP4_META_SW_CORRECT=0
  b grp_swc3_p'$PAD' -DNVFP4_META_GROUPED=1 -DNVFP4_META_GROUP_SIZE=8 -DNVFP4_META_SW_CORRECT=3
  wait
  chown '"$(id -u):$(id -g)"' /out/grp_*'
ls -la $EXP/bin/grp_*
