/***************************************************************************************************
 * NVFP4 micro-benchmark for NVIDIA Blackwell SM120 (RTX 50 / RTX PRO Blackwell)
 *
 *  Pipeline
 *  --------
 *   1. Generate BF16 A [M x K] and B [N x K]   (row-major, K contiguous)
 *   2. HAND-WRITTEN quantization kernels : BF16 -> NVFP4  (torchao "two level" recipe)
 *        per_tensor_scale       = amax / (448 * 6)
 *        block_scale            = block_amax(16) / 6
 *        stored E4M3 scale byte = quantize_e4m3( block_scale / per_tensor_scale )
 *        packed FP4 (E2M1, 2/B) = quantize_e2m1( x / (per_tensor_scale * dequant(E4M3 byte)) )
 *        scale bytes are written directly into CUTLASS' interleaved SF layout
 *   3. CUTLASS NVFP4 x NVFP4 -> BF16 GEMM on the FP4 Tensor Core (example 79a config)
 *        D = alpha * (Aq @ Bq^T),   alpha = per_tensor_scale_A * per_tensor_scale_B
 *   4. Correctness vs a BF16 cuBLAS reference built from the full-precision inputs
 *   5. Separate timing of the quantization phase and the GEMM kernel, plus a
 *      cudaProfilerApi + NVTX window around ONE instance of each, for Nsight Compute.
 *
 *  Kernels visible to Nsight Compute:
 *      fill_normal_bf16   amax_bf16   scale_from_amax   quantize_nvfp4   dequant_nvfp4
 *      diff_norms         cast_bf16_f32
 *      cutlass::device_kernel<...>   (the FP4 Tensor Core GEMM; target via NVTX range "gemm")
 *
 *  Build: build.sh   Run: run.sh   Profile: profile.sh
 **************************************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <string>
#include <utility>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_fp6.h>
#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>

// Largest-error-selection metadata config -- must be defined before collective_builder.hpp
// pulls in the patched sm120_blockscaled_mma_tma.hpp (its own #ifndef defaults are Linear
// mode). Override via -D from build.sh:
//   Linear (A@W^T):  NVFP4_META_GROUP_SIZE=4 NVFP4_META_ON_B=0 (default)
//   QK^T (attention): NVFP4_META_GROUP_SIZE=8 NVFP4_META_ON_B=1
#ifndef NVFP4_META_GROUP_SIZE
#define NVFP4_META_GROUP_SIZE 4
#endif
#ifndef NVFP4_META_ON_B
#define NVFP4_META_ON_B 0
#endif
// NVFP4_META_ENABLE=0 builds against an UNPATCHED CUTLASS checkout (no metaA/metaB
// TMA loading at all) for baseline (no-metadata) A/B timing comparisons.
#ifndef NVFP4_META_ENABLE
#define NVFP4_META_ENABLE 1
#endif

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/float_subbyte.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

#include "helper.h"   // CUDA_CHECK / CUTLASS_CHECK / GpuTimer

using namespace cute;

#define CUBLAS_CHECK(x)                                                          \
  do {                                                                          \
    cublasStatus_t s_ = (x);                                                    \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                          \
      std::fprintf(stderr, "cuBLAS error %d at line %d\n", (int)s_, __LINE__);  \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------------------------
// CUTLASS GEMM type  (identical to CUTLASS example 79a: nvfp4 x nvfp4 -> bf16, SM120)
// ---------------------------------------------------------------------------------------------
using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;   // {DataType=e2m1, ScaleFactorType=ue4m3}
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;                  // operand B is (N,K); ColumnMajor => K contiguous
constexpr int AlignmentB = 32;

using ElementD   = cutlass::bfloat16_t;
using ElementC   = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm120;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;

using ThreadBlockShape = Shape<_128,_128,_128>;
using ClusterShape     = Shape<_1,_1,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    // Extra NVFP4_SMEM_PAD-byte carveout: metaA/metaB (added directly in the patched
    // sm120_blockscaled_mma_tma.hpp) need to be pipelined across Stages just like
    // A/B/SFA/SFB, but the auto-carveout formula sizes Stages from the ORIGINAL
    // per-stage footprint (it has no way to know about the patch's addition) and
    // leaves no headroom for it. Padding the carveout drops Stages by exactly one,
    // freeing enough SMEM for metaA/metaB's own per-stage buffers. 25000 tuned
    // empirically via CUTLASS_DEBUG_TRACE_LEVEL=1 ("Setting smem size to ..." trace).
    //
    // IMPORTANT: this padding is shared by BOTH the patched (selection) and unpatched
    // (baseline) builds, since both include this same header. For the unpatched build
    // there's no metaA to make room for, so the padding has no purpose there -- but it
    // still forces CUTLASS to pick one fewer Stage than it naturally would (measured:
    // Stage 3 instead of Stage 4). This makes the two builds share the same Stage count
    // (good: no confound from differing pipeline depth) but means neither baseline nor
    // selection runs at ITS OWN natural optimum. A "true baseline" (-DNVFP4_SMEM_PAD=0,
    // built against unpatched CUTLASS) picks Stage 4 for this problem size -- empirically
    // found to be SLOWER than the shared Stage-3 config here (more stages isn't always
    // faster: SMEM pressure can reduce occupancy enough to offset better latency hiding).
    // See README.md "Stage-count confound" for the full investigation and numbers.
#ifndef NVFP4_SMEM_PAD
#define NVFP4_SMEM_PAD 25000
#endif
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage)) + NVFP4_SMEM_PAD>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA   = typename Gemm::GemmKernel::StrideA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

constexpr float F4_E2M1_MAX = 6.0f;
constexpr float F8_E4M3_MAX = 448.0f;

// nearest representable |E2M1| value, from the (already scaled) magnitude -- no dequant round-trip.
// levels: 0 .5 1 1.5 2 3 4 6 ; round-to-nearest thresholds at the midpoints.
__device__ __forceinline__ float e2m1_nearest_mag(float a) {
  if (a < 0.25f) return 0.f;
  if (a < 0.75f) return 0.5f;
  if (a < 1.25f) return 1.f;
  if (a < 1.75f) return 1.5f;
  if (a < 2.5f)  return 2.f;
  if (a < 3.5f)  return 3.f;
  if (a < 5.0f)  return 4.f;
  return 6.f;
}

// ---------------------------------------------------------------------------------------------
// Hand-written kernels
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float hash_unit(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return (x >> 8) * (1.0f / 16777216.0f);
}

__global__ void fill_normal_bf16(__nv_bfloat16* p, size_t n, uint32_t seed) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t s  = (uint32_t)(i * 2654435761ULL) ^ seed;
  float u1 = fmaxf(hash_unit(s), 1e-7f);
  float u2 = hash_unit(s * 747796405U + 2891336453U);
  float r  = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);   // ~N(0,1)
  p[i] = __float2bfloat16(r);
}

__global__ void cast_bf16_f32(const __nv_bfloat16* __restrict__ s, float* __restrict__ d, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    d[i] = __bfloat162float(s[i]);
}

// per-tensor amax via atomicMax on the int bit pattern (valid for non-negative floats)
__global__ void amax_bf16(const __nv_bfloat16* __restrict__ x, size_t n, float* __restrict__ out) {
  float m = 0.f;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    m = fmaxf(m, fabsf(__bfloat162float(x[i])));
  __shared__ float sm[32];
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffU, m, o));
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    int lanes = (blockDim.x + 31) >> 5;
    m = (threadIdx.x < lanes) ? sm[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffU, m, o));
    if (threadIdx.x == 0) atomicMax((int*)out, __float_as_int(m));
  }
}

__global__ void scale_from_amax(const float* __restrict__ amax, float* __restrict__ ps) {
  if (threadIdx.x == 0) *ps = (*amax) / (F8_E4M3_MAX * F4_E2M1_MAX);
}

// One thread per (row, 16-element K block): 8 packed FP4 bytes + 1 E4M3 scale byte.
// DST = destination for the quantized outputs:
//   0 = compute-only  : skip stores, fold into *sink        (fused, consumed straight from regs)
//   1 = DRAM          : global stores                        (separate quantize kernel)
//   2 = SMEM          : store to __shared__, fold back       (fused, staged in shared memory for the MMA)
template <int DST, class SfLayout>
__global__ void quantize_nvfp4(const __nv_bfloat16* __restrict__ x, int rows, int K,
                               const float* __restrict__ ps_ptr,
                               uint8_t* __restrict__ qdata, uint8_t* __restrict__ sf,
                               SfLayout sfl, uint32_t* __restrict__ sink) {
  __shared__ uint8_t s_q[128][8];
  __shared__ uint8_t s_sf[128];
  int kb  = blockIdx.x * blockDim.x + threadIdx.x;   // block index along K
  int row = blockIdx.y;
  if (row >= rows || kb >= (K >> 4)) return;
  const int lt = threadIdx.x;

  const __nv_bfloat16* xp = x + (size_t)row * K + (size_t)kb * 16;
  float v[16], bamax = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) { v[j] = __bfloat162float(xp[j]); bamax = fmaxf(bamax, fabsf(v[j])); }

  float ps     = *ps_ptr;
  float scaled = fminf(fmaxf((bamax / F4_E2M1_MAX) / ps, 1.0e-30f), F8_E4M3_MAX);
  __nv_fp8_storage_t sfb = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
  float total = ps * __half2float(__nv_cvt_fp8_to_halfraw(sfb, __NV_E4M3));
  float inv   = (total > 0.f) ? (1.0f / total) : 0.f;

  uint8_t* op = qdata + (size_t)row * (K / 2) + (size_t)kb * 8;
  uint32_t acc = 0;
#pragma unroll
  for (int j = 0; j < 16; j += 2) {
    float q0 = fminf(fmaxf(v[j]     * inv, -F4_E2M1_MAX), F4_E2M1_MAX);
    float q1 = fminf(fmaxf(v[j + 1] * inv, -F4_E2M1_MAX), F4_E2M1_MAX);
    // fp4x2: 1st component -> low nibble, 2nd -> high nibble (matches CUTLASS even-index-low)
    __nv_fp4x2_storage_t b = __nv_cvt_float2_to_fp4x2(make_float2(q0, q1), __NV_E2M1, cudaRoundNearest);
    if      (DST == 1) op[j >> 1]      = (uint8_t)b;
    else if (DST == 2) s_q[lt][j >> 1] = (uint8_t)b;
    else               acc ^= (uint32_t)(uint8_t)b << ((j & 6) << 2);
  }
  if (DST == 1) { sf[ sfl(row, kb * 16, 0) ] = sfb; return; }
  if (DST == 2) {
    s_sf[lt] = sfb;
    __syncwarp();
    uint32_t f = s_sf[lt];
#pragma unroll
    for (int k = 0; k < 8; ++k) f ^= (uint32_t)s_q[lt][k] << ((k & 3) << 3);
    if ((kb & 1023) == 0) atomicAdd(sink, f);
    return;
  }
  if ((kb & 1023) == 0) atomicAdd(sink, acc ^ sfb);
}

// Same as quantize_nvfp4, but additionally: for each group of NVFP4_META_GROUP_SIZE elements,
// find the one with the largest E2M1 quantization error and emit {index, 2 extra mantissa bits
// (from an E2M3 re-quantization)} into `meta` (1 byte per group). Group size/index width follow
// the same NVFP4_META_GROUP_SIZE macro as the CUTLASS collective patch (4 for Linear, 8 for QK^T).
// Used only to measure the added quantization cost of the largest-error selection.
// CHEAP_ERR=false : per-element error via E2M1 dequant round-trip (__nv_cvt_fp4_to_halfraw x16)
// CHEAP_ERR=true  : per-element error = |scaled - nearest_E2M1(scaled)| * total  (rounding residual,
//                   no dequant) -- the cheaper way to get the selection criterion.
template <int DST, bool CHEAP_ERR, class SfLayout>
__global__ void quantize_nvfp4_select(const __nv_bfloat16* __restrict__ x, int rows, int K,
                                       const float* __restrict__ ps_ptr,
                                       uint8_t* __restrict__ qdata, uint8_t* __restrict__ sf,
                                       uint8_t* __restrict__ meta, SfLayout sfl,
                                       uint32_t* __restrict__ sink) {
  __shared__ uint8_t s_q[128][8];
  __shared__ uint8_t s_sf[128];
  __shared__ uint8_t s_m[128][4];
  int kb  = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (row >= rows || kb >= (K >> 4)) return;
  const int lt = threadIdx.x;

  const __nv_bfloat16* xp = x + (size_t)row * K + (size_t)kb * 16;
  float v[16], bamax = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) { v[j] = __bfloat162float(xp[j]); bamax = fmaxf(bamax, fabsf(v[j])); }

  float ps     = *ps_ptr;
  float scaled = fminf(fmaxf((bamax / F4_E2M1_MAX) / ps, 1.0e-30f), F8_E4M3_MAX);
  __nv_fp8_storage_t sfb = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
  float total = ps * __half2float(__nv_cvt_fp8_to_halfraw(sfb, __NV_E4M3));
  float inv   = (total > 0.f) ? (1.0f / total) : 0.f;

  uint8_t* op = qdata + (size_t)row * (K / 2) + (size_t)kb * 8;
  uint32_t acc = 0;
  float err[16];
#pragma unroll
  for (int j = 0; j < 16; j += 2) {
    float q0 = fminf(fmaxf(v[j]     * inv, -F4_E2M1_MAX), F4_E2M1_MAX);
    float q1 = fminf(fmaxf(v[j + 1] * inv, -F4_E2M1_MAX), F4_E2M1_MAX);
    __nv_fp4x2_storage_t b = __nv_cvt_float2_to_fp4x2(make_float2(q0, q1), __NV_E2M1, cudaRoundNearest);
    if      (DST == 1 || DST == 3) op[j >> 1] = (uint8_t)b;   // DST==3: real qdata write, meta discarded below
    else if (DST == 2) s_q[lt][j >> 1] = (uint8_t)b;
    else               acc ^= (uint32_t)(uint8_t)b << ((j & 6) << 2);
    // per-element quantization error (the selection criterion)
    if (CHEAP_ERR) {
      // E2M1 rounding residual in the scaled domain; the common factor `total` is the same for
      // every element in the group, so it is omitted -- argmax is unchanged.
      float a0 = fabsf(q0), a1 = fabsf(q1);
      err[j]     = fabsf(a0 - e2m1_nearest_mag(a0));
      err[j + 1] = fabsf(a1 - e2m1_nearest_mag(a1));
    } else {
      float d0 = total * __half2float(__nv_cvt_fp4_to_halfraw((__nv_fp4_storage_t)((uint8_t)b & 0xF), __NV_E2M1));
      float d1 = total * __half2float(__nv_cvt_fp4_to_halfraw((__nv_fp4_storage_t)((uint8_t)b >> 4),  __NV_E2M1));
      err[j]     = fabsf(v[j]     - d0);
      err[j + 1] = fabsf(v[j + 1] - d1);
    }
  }

  constexpr int GS = NVFP4_META_GROUP_SIZE;          // 4 (Linear) or 8 (QK^T)
  constexpr int NG = 16 / GS;                         // groups per 16-elem chunk
  constexpr int IDXBITS = (GS == 8) ? 3 : 2;           // ceil(log2(GS)) for GS in {4,8}
  constexpr uint8_t IDXMASK = (uint8_t)(GS - 1);
  uint8_t* mp = meta + (size_t)row * (K / GS) + (size_t)kb * NG;
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    int base = g * GS;
    float emax = -1.f; int eidx = 0;
#pragma unroll
    for (int t = 0; t < GS; ++t) {
      if (err[base + t] > emax) { emax = err[base + t]; eidx = t; }
    }
    float qw = fminf(fmaxf(v[base + eidx] * inv, -F4_E2M1_MAX), F4_E2M1_MAX);
    __nv_fp6_storage_t e2m3 = __nv_cvt_float_to_fp6(qw, __NV_E2M3, cudaRoundNearest);
    uint8_t m = (uint8_t)(((uint8_t)(e2m3 & 0x3) << IDXBITS) | (eidx & IDXMASK));  // 2 extra mantissa bits + index
    if      (DST == 1) mp[g]       = m;
    else if (DST == 2) s_m[lt][g]  = m;
    else               acc ^= (uint32_t)m << (g << 3);   // DST==0 or 3: metadata discarded (sink only)
  }
  if (DST == 1 || DST == 3) { sf[ sfl(row, kb * 16, 0) ] = sfb; return; }   // DST==3: real sf write too
  if (DST == 2) {
    s_sf[lt] = sfb;
    __syncwarp();
    uint32_t f = s_sf[lt];
#pragma unroll
    for (int g = 0; g < NG; ++g) f ^= (uint32_t)s_m[lt][g] << ((g & 3) << 3);
#pragma unroll
    for (int k = 0; k < 8; ++k) f ^= (uint32_t)s_q[lt][k] << ((k & 3) << 3);
    if ((kb & 1023) == 0) atomicAdd(sink, f);
    return;
  }
  if ((kb & 1023) == 0) atomicAdd(sink, acc ^ sfb);
}

// Models the GEMM kernel's OWN TMA prologue reading the selection metadata back from
// DRAM -- separate kernel launch from quantize_nvfp4_select, since that's how a real
// producer-epilogue -> consumer-GEMM handoff works (SMEM does not persist across
// kernel launches, so "SMEM store" above only models a fully fused single kernel; this
// instead measures the extra load the stock CUTLASS GEMM used here doesn't do, since
// it has no metadata operand at all). Plain bandwidth-bound grid-stride read, same
// shape as what nvfp4_mma_atom.cu's TMA load of metaA/metaB actually issues.
__global__ void reload_meta(const uint8_t* __restrict__ meta, size_t n, uint32_t* __restrict__ sink) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  uint32_t acc = 0;
  for (; i < n; i += stride) acc ^= meta[i];
  if (acc == 0x12345678u) atomicAdd(sink, acc);   // never true; keeps the load live
}

// ---- real fusion experiment: is quantization actually fused into a producer kernel's
// epilogue, with NO BF16 ever touching DRAM, vs. that producer just writing BF16 out
// normally? No subtraction/approximation -- both kernels below compute the SAME 16
// synthetic "activation" values entirely in registers (standing in for whatever a real
// epilogue -- RMSNorm, SwiGLU combine, an activation function -- would have just
// finished computing) and only differ in what they do with them once ready to write.

// Cheap, deterministic, DRAM-read-free per-element value -- a stand-in for "whatever
// the producing kernel just computed in its own registers." Real epilogues (RMSNorm,
// SwiGLU, GELU) are a few FLOPs per output element and are themselves not the subject
// of this measurement; only the marginal cost of what happens to the result matters.
__device__ __forceinline__ float synth_val(size_t idx) {
  uint32_t h = (uint32_t)idx * 2654435761u;
  float u = __uint_as_float(0x3f800000u | (h & 0x007fffffu)) - 1.0f;   // in [0,1)
  return (u - 0.5f) * 4.0f;                                            // in [-2,2)
}

// Quantize+store one 16-element group, no selection (mirrors quantize_nvfp4's DST==1 body).
__device__ __forceinline__
void quantize_group16(const float v[16], float ps, uint8_t* op, uint8_t* sfp) {
  float bamax = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) bamax = fmaxf(bamax, fabsf(v[j]));
  float scaled = fminf(fmaxf((bamax / F4_E2M1_MAX) / ps, 1.0e-30f), F8_E4M3_MAX);
  __nv_fp8_storage_t sfb = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
  float total = ps * __half2float(__nv_cvt_fp8_to_halfraw(sfb, __NV_E4M3));
  float inv   = (total > 0.f) ? (1.0f / total) : 0.f;
#pragma unroll
  for (int j = 0; j < 16; j += 2) {
    float q0 = fminf(fmaxf(v[j]     * inv, -F4_E2M1_MAX), F4_E2M1_MAX);
    float q1 = fminf(fmaxf(v[j + 1] * inv, -F4_E2M1_MAX), F4_E2M1_MAX);
    __nv_fp4x2_storage_t b = __nv_cvt_float2_to_fp4x2(make_float2(q0, q1), __NV_E2M1, cudaRoundNearest);
    op[j >> 1] = (uint8_t)b;
  }
  *sfp = sfb;
}

// Quantize+select+store one 16-element group, no DST branching -- always a real DRAM
// store (mirrors quantize_nvfp4_select's DST==1 body).
template <bool CHEAP_ERR>
__device__ __forceinline__
void quantize_select_group16(const float v[16], float ps, uint8_t* op, uint8_t* sfp, uint8_t* mp) {
  float bamax = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) bamax = fmaxf(bamax, fabsf(v[j]));
  float scaled = fminf(fmaxf((bamax / F4_E2M1_MAX) / ps, 1.0e-30f), F8_E4M3_MAX);
  __nv_fp8_storage_t sfb = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
  float total = ps * __half2float(__nv_cvt_fp8_to_halfraw(sfb, __NV_E4M3));
  float inv   = (total > 0.f) ? (1.0f / total) : 0.f;
  float err[16];
#pragma unroll
  for (int j = 0; j < 16; j += 2) {
    float q0 = fminf(fmaxf(v[j]     * inv, -F4_E2M1_MAX), F4_E2M1_MAX);
    float q1 = fminf(fmaxf(v[j + 1] * inv, -F4_E2M1_MAX), F4_E2M1_MAX);
    __nv_fp4x2_storage_t b = __nv_cvt_float2_to_fp4x2(make_float2(q0, q1), __NV_E2M1, cudaRoundNearest);
    op[j >> 1] = (uint8_t)b;
    if (CHEAP_ERR) {
      float a0 = fabsf(q0), a1 = fabsf(q1);
      err[j]     = fabsf(a0 - e2m1_nearest_mag(a0));
      err[j + 1] = fabsf(a1 - e2m1_nearest_mag(a1));
    } else {
      float d0 = total * __half2float(__nv_cvt_fp4_to_halfraw((__nv_fp4_storage_t)((uint8_t)b & 0xF), __NV_E2M1));
      float d1 = total * __half2float(__nv_cvt_fp4_to_halfraw((__nv_fp4_storage_t)((uint8_t)b >> 4),  __NV_E2M1));
      err[j]     = fabsf(v[j]     - d0);
      err[j + 1] = fabsf(v[j + 1] - d1);
    }
  }
  *sfp = sfb;
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    int base = g * 4;
    float emax = -1.f; int eidx = 0;
#pragma unroll
    for (int t = 0; t < 4; ++t) if (err[base + t] > emax) { emax = err[base + t]; eidx = t; }
    float qw = fminf(fmaxf(v[base + eidx] * inv, -F4_E2M1_MAX), F4_E2M1_MAX);
    __nv_fp6_storage_t e2m3 = __nv_cvt_float_to_fp6(qw, __NV_E2M3, cudaRoundNearest);
    mp[g] = (uint8_t)(((uint8_t)(e2m3 & 0x3) << 2) | (eidx & 0x3));
  }
}

// Baseline producer: computes 16 synthetic values, writes them out as BF16 -- an
// ordinary epilogue's normal (unquantized) output write. No selection, no quantization.
__global__ void producer_bf16(int rows, int K, __nv_bfloat16* __restrict__ out) {
  int kb  = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (row >= rows || kb >= (K >> 4)) return;
  size_t base = (size_t)row * K + (size_t)kb * 16;
  // vectorized 16-byte stores (2x uint4, 8 bf16 each) -- 16 scalar 2-byte stores here
  // would badly under-report real bandwidth and unfairly inflate the apparent fusion
  // win, since a real epilogue's BF16 write is vectorized too.
  uint32_t w[8];
#pragma unroll
  for (int j = 0; j < 16; j += 2) {
    uint16_t lo = __bfloat16_as_ushort(__float2bfloat16(synth_val(base + j)));
    uint16_t hi = __bfloat16_as_ushort(__float2bfloat16(synth_val(base + j + 1)));
    w[j >> 1] = (uint32_t)lo | ((uint32_t)hi << 16);
  }
  uint4* o4 = reinterpret_cast<uint4*>(out + base);
  o4[0] = make_uint4(w[0], w[1], w[2], w[3]);
  o4[1] = make_uint4(w[4], w[5], w[6], w[7]);
}

// Fused producer, quantize only (no selection): SAME synthetic values as producer_bf16,
// but quantized straight out of registers -- the BF16 value never touches DRAM.
template <class SfLayout>
__global__ void producer_fused_quant(int rows, int K, float ps,
                                     uint8_t* __restrict__ qdata, uint8_t* __restrict__ sf,
                                     SfLayout sfl) {
  int kb  = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (row >= rows || kb >= (K >> 4)) return;
  size_t base = (size_t)row * K + (size_t)kb * 16;
  float v[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) v[j] = synth_val(base + j);
  uint8_t* op  = qdata + (size_t)row * (K / 2) + (size_t)kb * 8;
  uint8_t* sfp = sf + sfl(row, kb * 16, 0);
  quantize_group16(v, ps, op, sfp);
}

// Fused producer, quantize + selection: SAME synthetic values, fused quantize+select
// straight out of registers, BF16 never touches DRAM.
template <bool CHEAP_ERR, class SfLayout>
__global__ void producer_fused_select(int rows, int K, float ps,
                                      uint8_t* __restrict__ qdata, uint8_t* __restrict__ sf,
                                      uint8_t* __restrict__ meta, SfLayout sfl) {
  int kb  = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (row >= rows || kb >= (K >> 4)) return;
  size_t base = (size_t)row * K + (size_t)kb * 16;
  float v[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) v[j] = synth_val(base + j);
  uint8_t* op  = qdata + (size_t)row * (K / 2) + (size_t)kb * 8;
  uint8_t* sfp = sf + sfl(row, kb * 16, 0);
  uint8_t* mp  = meta + (size_t)row * (K / 4) + (size_t)kb * 4;
  quantize_select_group16<CHEAP_ERR>(v, ps, op, sfp, mp);
}

template <class SfLayout>
__global__ void dequant_nvfp4(const uint8_t* __restrict__ qdata, const uint8_t* __restrict__ sf,
                              SfLayout sfl, const float* __restrict__ ps_ptr,
                              int rows, int K, __nv_bfloat16* __restrict__ out) {
  int kb  = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (row >= rows || kb >= (K >> 4)) return;

  float total = (*ps_ptr) *
      __half2float(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)sf[sfl(row, kb * 16, 0)], __NV_E4M3));
  const uint8_t* op = qdata + (size_t)row * (K / 2) + (size_t)kb * 8;
  __nv_bfloat16* o  = out + (size_t)row * K + (size_t)kb * 16;
#pragma unroll
  for (int j = 0; j < 16; j += 2) {
    uint8_t b = op[j >> 1];
    float d0 = __half2float(__nv_cvt_fp4_to_halfraw((__nv_fp4_storage_t)(b & 0xF), __NV_E2M1));
    float d1 = __half2float(__nv_cvt_fp4_to_halfraw((__nv_fp4_storage_t)(b >> 4),  __NV_E2M1));
    o[j]     = __float2bfloat16(total * d0);
    o[j + 1] = __float2bfloat16(total * d1);
  }
}

// acc[0] += sum((bf16 d - f32 ref)^2)   acc[1] += sum(ref^2)
__global__ void diff_norms(const __nv_bfloat16* __restrict__ d, const float* __restrict__ ref,
                           size_t n, double* __restrict__ acc) {
  double e = 0.0, r = 0.0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    double dv = (double)__bfloat162float(d[i]);
    double rv = (double)ref[i];
    e += (dv - rv) * (dv - rv);
    r += rv * rv;
  }
  __shared__ double se[256], sr[256];
  se[threadIdx.x] = e; sr[threadIdx.x] = r;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) { se[threadIdx.x] += se[threadIdx.x + s]; sr[threadIdx.x] += sr[threadIdx.x + s]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { atomicAdd(&acc[0], se[0]); atomicAdd(&acc[1], sr[0]); }
}

static double rel_frob(const __nv_bfloat16* d, const float* ref, size_t n) {
  double* dacc = nullptr;
  CUDA_CHECK(cudaMalloc(&dacc, 2 * sizeof(double)));
  CUDA_CHECK(cudaMemset(dacc, 0, 2 * sizeof(double)));
  diff_norms<<<1024, 256>>>(d, ref, n, dacc);
  double acc[2];
  CUDA_CHECK(cudaMemcpy(acc, dacc, 2 * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(dacc));
  return std::sqrt(acc[0]) / std::sqrt(acc[1]);
}

// ---------------------------------------------------------------------------------------------
