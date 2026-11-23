#!/bin/bash
# Build the FP4 OMMA kernels for the RTX PRO 4000 Blackwell (sm_120a) and dump SASS.
# compute_120a/sm_120a explicitly: plain sm_120 PTX rejects .kind::mxf4nvf4/.block_scale.
set -e
source "$(dirname "$0")/env.sh"
mkdir -p "$EXP/bin" "$EXP/sass"
for n in 1 4; do
  nvcc -std=c++17 -O3 -lineinfo -gencode=arch=compute_120a,code=sm_120a -DNCHAIN=$n \
       "$EXP/src/omma_fp4.cu" -o "$EXP/bin/omma_fp4_c$n"
  cuobjdump -sass "$EXP/bin/omma_fp4_c$n" > "$EXP/sass/omma_fp4_c$n.cuobjdump.sass"
  cuobjdump -xelf all "$EXP/bin/omma_fp4_c$n" >/dev/null 2>&1 || true
done
# Unmodified CUTLASS 79a (NVFP4 x NVFP4 -> BF16, sm_120 block-scaled TMA collective), from a
# clean `git archive 147295a3` of ~/workspace/cutlass (its working tree has local edits to
# sm120_blockscaled_mma_tma.hpp that are NOT used here).
C=$EXP/cutlass_clean
nvcc -std=c++17 -O3 -gencode=arch=compute_120a,code=sm_120a --expt-relaxed-constexpr -lineinfo \
     -I$C/include -I$C/tools/util/include -I$C/examples/common \
     $C/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu -o $EXP/bin/cutlass_79a_nvfp4_gemm
cuobjdump -sass $EXP/bin/cutlass_79a_nvfp4_gemm > $EXP/sass/cutlass_79a.cuobjdump.sass
