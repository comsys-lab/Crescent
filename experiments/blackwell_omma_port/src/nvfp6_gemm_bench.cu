/*
 * Block-scaled GEMM format comparison on SM120 (pristine CUTLASS, GEMM time only):
 *   FMT=0  NVFP4 (A) x NVFP4 (B)   e2m1, ue4m3 scale / 16  (kind::mxf4nvf4)
 *   FMT=1  MXFP4 (A) x MXFP4 (B)   e2m1, ue8m0 scale / 32
 *   FMT=2  MXFP6 (A) x MXFP4 (B)   e2m3 activation, e2m1 weight, ue8m0 / 32 (kind::mxf8f6f4)
 *   FMT=3  MXFP6 (A) x MXFP6 (B)   e2m3 x e2m3 (QK^T of the NVFP6 comparison), ue8m0 / 32
 * (copied from nvfp4_microbench/code/mx_bench.cu; FMT=3 added)
 * CZ=1 (build against cutlass_mode3 with the NVFP4_META_* macros): Czardas correction data passed to the
 *   patched collective -- Linear: NVFP4_META_PREPACK (M/2, K/4) bytes on A; QK^T: NVFP4_META_TRANSPOSE
 *   (M-contiguous (M, K/16)) on A and B. Random bytes (timing only).
 * Operands are random bit patterns (finite for these formats); scales are fixed to 1.0.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#ifndef FMT
#define FMT 0
#endif
#ifndef CZ
#define CZ 0
#endif
#ifndef NVFP4_SMEM_PAD
#define NVFP4_SMEM_PAD 0   // extra carveout: equal Stage count with / without the extra TMA tensor
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
#include "helper.h"

using namespace cute;

#if FMT == 0
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
constexpr int AlignmentA = 32, AlignmentB = 32;
constexpr uint8_t kSfOne = 0x38;                 // ue4m3 1.0
static const char* kName = "nvfp4xnvfp4";
#elif FMT == 1
using ElementA = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
constexpr int AlignmentA = 32, AlignmentB = 32;
constexpr uint8_t kSfOne = 127;                  // ue8m0 1.0
static const char* kName = "mxfp4xmxfp4";
#elif FMT == 3
using ElementA = cutlass::mx_float6_t<cutlass::float_e2m3_t>;
using ElementB = cutlass::mx_float6_t<cutlass::float_e2m3_t>;
constexpr int AlignmentA = 128, AlignmentB = 128;
constexpr uint8_t kSfOne = 127;
static const char* kName = "mxfp6xmxfp6";
#else
using ElementA = cutlass::mx_float6_t<cutlass::float_e2m3_t>;
using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
constexpr int AlignmentA = 128, AlignmentB = 128;
constexpr uint8_t kSfOne = 127;
static const char* kName = "mxfp6xmxfp4";
#endif

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using ElementD   = cutlass::bfloat16_t;
using ElementC   = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm120;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape   = Shape<_128,_128,_128>;
using ClusterShape       = Shape<_1,_1,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage)) + NVFP4_SMEM_PAD>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

__global__ void fill_random_bytes(uint8_t* p, size_t n, uint32_t seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)(i * 2654435761ULL) ^ seed;
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    p[i] = (uint8_t)x;
  }
}

int main(int argc, char** argv) {
  int M = 4096, N = 4096, K = 4096, L = 1, iters = 20, warmup = 3;
  printf("STAGES %d\n", (int)Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Stages);
  for (int i = 1; i < argc; ++i) {
    if      (!strncmp(argv[i], "--m=", 4)) M = atoi(argv[i] + 4);
    else if (!strncmp(argv[i], "--n=", 4)) N = atoi(argv[i] + 4);
    else if (!strncmp(argv[i], "--k=", 4)) K = atoi(argv[i] + 4);
    else if (!strncmp(argv[i], "--l=", 4)) L = atoi(argv[i] + 4);
    else if (!strncmp(argv[i], "--iters=", 8)) iters = atoi(argv[i] + 8);
    else if (!strncmp(argv[i], "--warmup=", 9)) warmup = atoi(argv[i] + 9);
  }
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, L});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, L});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, L});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, L});
  LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, L));
  LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, L));

  const size_t bytesA = (size_t)M * K * L * cutlass::sizeof_bits<typename ElementA::DataType>::value / 8;
  const size_t bytesB = (size_t)N * K * L * cutlass::sizeof_bits<typename ElementB::DataType>::value / 8;
  const size_t bytesSFA = (size_t)size(filter_zeros(layout_SFA)), bytesSFB = (size_t)size(filter_zeros(layout_SFB));
  uint8_t *dA, *dB, *dSFA, *dSFB; ElementD* dD;
#if CZ
#if NVFP4_META_PREPACK
  auto stride_MetaA = cutlass::make_cute_packed_stride(StrideA{}, {M / 2, K / 4, L});
#elif NVFP4_META_TRANSPOSE
  auto stride_MetaA = cute::make_stride(cute::Int<1>{}, (int64_t)M, (int64_t)M * (K / NVFP4_META_GROUP_SIZE));
#else
  auto stride_MetaA = cutlass::make_cute_packed_stride(StrideA{}, {M, K / NVFP4_META_GROUP_SIZE, L});
#endif
  uint8_t* dMetaA; CUDA_CHECK(cudaMalloc(&dMetaA, (size_t)M * K / NVFP4_META_GROUP_SIZE * L));
  fill_random_bytes<<<1024, 256>>>(dMetaA, (size_t)M * K / NVFP4_META_GROUP_SIZE * L, 0x55u);
#if NVFP4_META_ON_B
  auto stride_MetaB = cute::make_stride(cute::Int<1>{}, (int64_t)N, (int64_t)N * (K / NVFP4_META_GROUP_SIZE));
  uint8_t* dMetaB; CUDA_CHECK(cudaMalloc(&dMetaB, (size_t)N * K / NVFP4_META_GROUP_SIZE * L));
  fill_random_bytes<<<1024, 256>>>(dMetaB, (size_t)N * K / NVFP4_META_GROUP_SIZE * L, 0x77u);
#endif
#endif
  CUDA_CHECK(cudaMalloc(&dA, bytesA)); CUDA_CHECK(cudaMalloc(&dB, bytesB));
  CUDA_CHECK(cudaMalloc(&dSFA, bytesSFA)); CUDA_CHECK(cudaMalloc(&dSFB, bytesSFB));
  CUDA_CHECK(cudaMalloc(&dD, (size_t)M * N * L * sizeof(ElementD)));
  fill_random_bytes<<<1024, 256>>>(dA, bytesA, 0x1234u);
  fill_random_bytes<<<1024, 256>>>(dB, bytesB, 0x9abcu);
  CUDA_CHECK(cudaMemset(dSFA, kSfOne, bytesSFA));
  CUDA_CHECK(cudaMemset(dSFB, kSfOne, bytesSFB));

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, L},
    { reinterpret_cast<typename ElementA::DataType const*>(dA), stride_A,
      reinterpret_cast<typename ElementB::DataType const*>(dB), stride_B,
      reinterpret_cast<typename ElementA::ScaleFactorType const*>(dSFA), layout_SFA,
      reinterpret_cast<typename ElementB::ScaleFactorType const*>(dSFB), layout_SFB
#if CZ
      , dMetaA, stride_MetaA
#if NVFP4_META_ON_B
      , dMetaB, stride_MetaB
#endif
#endif
    },
    { {1.f, 0.f}, reinterpret_cast<ElementC const*>(dD), stride_C, dD, stride_D } };
  Gemm gemm;
  cutlass::device_memory::allocation<uint8_t> ws(Gemm::get_workspace_size(args));
  CUTLASS_CHECK(gemm.can_implement(args));
  CUTLASS_CHECK(gemm.initialize(args, ws.get()));
  for (int i = 0; i < warmup; ++i) CUTLASS_CHECK(gemm.run());
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < iters; ++i) CUTLASS_CHECK(gemm.run());
  cudaEventRecord(e1); CUDA_CHECK(cudaEventSynchronize(e1)); CUDA_CHECK(cudaGetLastError());
  float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); ms /= iters;
  printf("RESULT fmt=%s M=%d N=%d K=%d L=%d gemm_ms=%.5f tflops=%.1f\n", kName, M, N, K, L, ms, 2.0 * M * N * K * L / (ms * 1e-3) / 1e12);
  return 0;
}
