// Activation quantize cost, E2M1 vs E2M3 codes, NV scaling (E4M3 / 16) and MX scaling (UE8M0 / 32),
// same kernel structure (MX: the two 16-element halves of a 32-block combine amax via one shuffle):
// one thread per (row, 16-element block): amax -> E4M3 block scale (x per-tensor scale) -> convert
// -> store packed codes (E2M1: 8 B / 16 elements, E2M3: 12 B / 16 elements) + 1 B scale.
// BF16 input M x K read from DRAM, codes + scales written to DRAM. Prints both times (ms).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp6.h>
#include <cuda_fp8.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

template <bool FP6, bool MX>
__global__ void quant(const __nv_bfloat16* __restrict__ x, int K, float ps, uint8_t* __restrict__ q, uint8_t* __restrict__ sf) {
  int kb = blockIdx.x * blockDim.x + threadIdx.x; size_t row = blockIdx.y;
  if (kb >= K / 16) return;
  const uint4* xp = reinterpret_cast<const uint4*>(x + row * K + (size_t)kb * 16);
  uint4 r0 = xp[0], r1 = xp[1];
  const __nv_bfloat16* h0 = reinterpret_cast<const __nv_bfloat16*>(&r0);
  const __nv_bfloat16* h1 = reinterpret_cast<const __nv_bfloat16*>(&r1);
  float v[16], am = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) { v[j] = __bfloat162float(h0[j]); v[8 + j] = __bfloat162float(h1[j]); }
#pragma unroll
  for (int j = 0; j < 16; ++j) am = fmaxf(am, fabsf(v[j]));
  const float qmax = FP6 ? 7.5f : 6.0f;
  float inv;
  if (MX) {
    am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 1));                 // 32-element block
    int e = am > 0.f ? (int)ceilf(log2f(am / qmax)) : -127;               // UE8M0: power-of-two scale
    e = max(-127, min(127, e));
    inv = exp2f((float)-e);
    if ((kb & 1) == 0) sf[row * (K / 32) + kb / 2] = (uint8_t)(e + 127);
  } else {
    float s = fminf(fmaxf(am / qmax / ps, 1e-30f), 448.f);
    __nv_fp8_storage_t sb = __nv_cvt_float_to_fp8(s, __NV_SATFINITE, __NV_E4M3);
    float t = ps * __half2float(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3)); inv = t > 0.f ? 1.f / t : 0.f;
    sf[row * (K / 16) + kb] = sb;
  }
  if (FP6) {
    uint32_t w[3] = {0, 0, 0};
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      uint32_t c = __nv_cvt_float_to_fp6(v[j] * inv, __NV_E2M3, cudaRoundNearest) & 0x3F;
      int bit = 6 * j; w[bit / 32] |= c << (bit % 32); if (bit % 32 > 26) w[bit / 32 + 1] |= c >> (32 - bit % 32);
    }
    uint32_t* o = reinterpret_cast<uint32_t*>(q + row * (K / 16 * 12) + (size_t)kb * 12);
    o[0] = w[0]; o[1] = w[1]; o[2] = w[2];
  } else {
    uint32_t w[2] = {0, 0};
#pragma unroll
    for (int j = 0; j < 16; j += 2) {
      uint32_t b = __nv_cvt_float2_to_fp4x2(make_float2(v[j] * inv, v[j + 1] * inv), __NV_E2M1, cudaRoundNearest);
      w[j / 8] |= b << ((j % 8) * 4);
    }
    *reinterpret_cast<uint2*>(q + row * (K / 2) + (size_t)kb * 8) = make_uint2(w[0], w[1]);
  }
}

int main(int argc, char** argv) {
  int M = 4096, K = 4096, iters = 50;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--m=", 4)) M = atoi(argv[i] + 4);
    else if (!strncmp(argv[i], "--k=", 4)) K = atoi(argv[i] + 4);
  }
  __nv_bfloat16* x; uint8_t *q, *sf;
  CK(cudaMalloc(&x, (size_t)M * K * 2)); CK(cudaMalloc(&q, (size_t)M * K)); CK(cudaMalloc(&sf, (size_t)M * K / 16));
  CK(cudaMemset(x, 0x3c, (size_t)M * K * 2));
  // M may exceed the 65535 grid-y limit: launch in row chunks
  auto launch = [&](int fp6) {
    for (int r0 = 0; r0 < M; r0 += 65535) {
      int rows = M - r0 < 65535 ? M - r0 : 65535;
      dim3 g((K / 16 + 127) / 128, rows);
      uint8_t* q6 = q + (size_t)r0 * (K / 16 * 12); uint8_t* q4 = q + (size_t)r0 * (K / 2); uint8_t* s0 = sf + (size_t)r0 * (K / 16);
      switch (fp6) {
        case 0: quant<false, false><<<g, 128>>>(x + (size_t)r0 * K, K, 1.f, q4, s0); break;
        case 1: quant<true,  false><<<g, 128>>>(x + (size_t)r0 * K, K, 1.f, q6, s0); break;
        case 2: quant<false, true ><<<g, 128>>>(x + (size_t)r0 * K, K, 1.f, q4, s0); break;
        case 3: quant<true,  true ><<<g, 128>>>(x + (size_t)r0 * K, K, 1.f, q6, s0); break;
      }
    }
  };
  float res[4];
  for (int f = 0; f < 4; ++f) {
    for (int i = 0; i < 5; ++i) launch(f);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); for (int i = 0; i < iters; ++i) launch(f); cudaEventRecord(b);
    CK(cudaEventSynchronize(b)); CK(cudaGetLastError());
    cudaEventElapsedTime(&res[f], a, b); res[f] /= iters;
  }
  printf("RESULT M=%d K=%d nv_e2m1_ms=%.5f nv_e2m3_ms=%.5f mx_e2m1_ms=%.5f mx_e2m3_ms=%.5f\n", M, K, res[0], res[1], res[2], res[3]);
  return 0;
}
