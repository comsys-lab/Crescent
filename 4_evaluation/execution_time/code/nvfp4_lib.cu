// NVFP4 Linear (quantize A -> CUTLASS NVFP4 GEMM) as a C API for end-to-end model forwards.
// NVFP4_META_ENABLE=0: unpatched CUTLASS, plain quantize (baseline). =1: selection quantize + metaA TMA load.
#ifndef NVFP4_META_ENABLE
#define NVFP4_META_ENABLE 0
#endif
#include "core.inc"
#include <vector>

struct Layer { int N, K, dev; uint8_t *Bq, *SFB; LayoutSFB lsfb; float psB; };
struct Dev { uint8_t *Aq=0, *SFA=0, *Ws=0, *Meta=0; size_t WsBytes=0; float *PsA=0, *Amax=0, *Alpha=0, *Tmp=0; uint32_t* Sink=0; bool up=false; };
static std::vector<Layer> g_layers;
static Dev g_dev[16];
static float g_psA_host = 0.05f;

__global__ void set_alpha(const float* psA, float psB, float* out) { if (threadIdx.x == 0) *out = (*psA) * psB; }

extern "C" int nvfp4_init(int maxM, int maxK) {
  int d; cudaGetDevice(&d); Dev& D = g_dev[d]; if (D.up) return 0;
  size_t sfa = (size_t)size(filter_zeros(Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(maxM, 128, maxK, 1))));
  CUDA_CHECK(cudaMalloc(&D.Aq, (size_t)maxM * maxK / 2));
  CUDA_CHECK(cudaMalloc(&D.SFA, sfa)); CUDA_CHECK(cudaMemset(D.SFA, 0, sfa));
  CUDA_CHECK(cudaMalloc(&D.PsA, 4)); CUDA_CHECK(cudaMalloc(&D.Amax, 4)); CUDA_CHECK(cudaMalloc(&D.Alpha, 4)); CUDA_CHECK(cudaMalloc(&D.Tmp, 4));
  CUDA_CHECK(cudaMalloc(&D.Sink, 4));
  CUDA_CHECK(cudaMalloc(&D.Meta, (size_t)maxM * maxK / 4));
  CUDA_CHECK(cudaMemcpy(D.PsA, &g_psA_host, 4, cudaMemcpyHostToDevice));
  D.up = true; return 0;
}
extern "C" void nvfp4_set_act_scale(float ps) {
  g_psA_host = ps; int d; cudaGetDevice(&d);
  CUDA_CHECK(cudaMemcpy(g_dev[d].PsA, &ps, 4, cudaMemcpyHostToDevice));
}

extern "C" int nvfp4_prepare(const void* W, int N, int K) {
  int d; cudaGetDevice(&d); Dev& D = g_dev[d];
  Layer L; L.N = N; L.K = K; L.dev = d;
  L.lsfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(128, N, K, 1));
  size_t sfb = (size_t)size(filter_zeros(L.lsfb));
  CUDA_CHECK(cudaMalloc(&L.Bq, (size_t)N * K / 2));
  CUDA_CHECK(cudaMalloc(&L.SFB, sfb)); CUDA_CHECK(cudaMemset(L.SFB, 0, sfb));
  CUDA_CHECK(cudaMemset(D.Amax, 0, 4));
  amax_bf16<<<1024, 256>>>((const __nv_bfloat16*)W, (size_t)N * K, D.Amax);
  scale_from_amax<<<1, 1>>>(D.Amax, D.Tmp);
  dim3 blk(128), grd((K / 16 + 127) / 128, N);
  quantize_nvfp4<1><<<grd, blk>>>((const __nv_bfloat16*)W, N, K, D.Tmp, L.Bq, L.SFB, L.lsfb, D.Sink);
  CUDA_CHECK(cudaMemcpy(&L.psB, D.Tmp, 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  g_layers.push_back(L);
  return (int)g_layers.size() - 1;
}

// out[M,N] bf16 = quant(x[M,K]) @ Wq^T. dynamic_amax=1: per-call activation amax (real dynamic scaling).
extern "C" int nvfp4_linear(int id, const void* x, int M, void* out, void* stream_, int dynamic_amax) {
  Layer& L = g_layers[id]; int N = L.N, K = L.K;
  CUDA_CHECK(cudaSetDevice(L.dev)); Dev& D = g_dev[L.dev];
  cudaStream_t s = (cudaStream_t)stream_;
  if (M == 0) return 0;
  LayoutSFA lsfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  if (dynamic_amax) {
    cudaMemsetAsync(D.Amax, 0, 4, s);
    amax_bf16<<<1024, 256, 0, s>>>((const __nv_bfloat16*)x, (size_t)M * K, D.Amax);
    scale_from_amax<<<1, 1, 0, s>>>(D.Amax, D.PsA);
  }
  set_alpha<<<1, 1, 0, s>>>(D.PsA, L.psB, D.Alpha);
  dim3 blk(128), grd((K / 16 + 127) / 128, M);
  // NVFP4_QUANTIZE_MODE (independent of NVFP4_META_ENABLE, which governs the GEMM/Gemm-type
  // side): 0 = match NVFP4_META_ENABLE (existing behavior); 1 = real selection SEARCH + CORRECTION
  // DATA COMPUTATION (DST=0, discarded -- no metadata write, no metaA alloc/stride) paired with
  // whatever GEMM NVFP4_META_ENABLE selects (intended: =0, true-base GEMM) -- isolates the real
  // end-to-end cost of adding JUST the selection computation, with no data-movement change at all.
#ifndef NVFP4_QUANTIZE_MODE
#define NVFP4_QUANTIZE_MODE 0
#endif
#if NVFP4_QUANTIZE_MODE == 1
  static_assert(NVFP4_META_ENABLE == 0, "compute-only quantize mode pairs with the true-base (unpatched) GEMM");
  // DST==3: qdata + scale factor are written FOR REAL (identical to the true-base quantize_nvfp4<1>
  // call below) -- only the metadata (selection/correction result) is discarded. Isolates the real
  // cost of the selection SEARCH + CORRECTION computation with zero change to memory traffic.
  quantize_nvfp4_select<3, false><<<grd, blk, 0, s>>>((const __nv_bfloat16*)x, M, K, D.PsA, D.Aq, D.SFA, D.Meta, lsfa, D.Sink);
#elif NVFP4_META_ENABLE
  quantize_nvfp4_select<1, false><<<grd, blk, 0, s>>>((const __nv_bfloat16*)x, M, K, D.PsA, D.Aq, D.SFA, D.Meta, lsfa, D.Sink);
  auto stride_MetaA = cutlass::make_cute_packed_stride(StrideA{}, {M, K / NVFP4_META_GROUP_SIZE, 1});
#else
  quantize_nvfp4<1><<<grd, blk, 0, s>>>((const __nv_bfloat16*)x, M, K, D.PsA, D.Aq, D.SFA, lsfa, D.Sink);
#endif
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
    { reinterpret_cast<ElementA::DataType const*>(D.Aq), stride_A,
      reinterpret_cast<ElementB::DataType const*>(L.Bq), stride_B,
      reinterpret_cast<ElementA::ScaleFactorType const*>(D.SFA), lsfa,
      reinterpret_cast<ElementB::ScaleFactorType const*>(L.SFB), L.lsfb
#if NVFP4_META_ENABLE
      , D.Meta, stride_MetaA
#endif
    },
    { {}, reinterpret_cast<ElementC const*>(out), stride_C, reinterpret_cast<ElementD*>(out), stride_D } };
  args.epilogue.thread.alpha_ptr = D.Alpha; args.epilogue.thread.beta = 0.f;
  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  if (ws > D.WsBytes) { if (D.Ws) cudaFree(D.Ws); CUDA_CHECK(cudaMalloc(&D.Ws, ws)); D.WsBytes = ws; }
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 1;
  if (gemm.initialize(args, D.Ws, s) != cutlass::Status::kSuccess) return 2;
  if (gemm.run(s) != cutlass::Status::kSuccess) return 3;
  return 0;
}
