// Minimal native Blackwell (sm_120a) block-scaled FP4 MMA -> SASS OMMA.
//
//   D[16x8] (f32) = sum_k A[16x64](e2m1) * SFA(ue4m3, 1 per 16 k)
//                         * B[8x64](e2m1)  * SFB(ue4m3, 1 per 16 k)  + C
//
// PTX: mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
//      .row.col.f32.e2m1.e2m1.f32.ue4m3   (NVFP4: 16-element blocks, UE4M3 scales)
//
// Plain ld.global fragment loads only (no TMA / mbarrier / ldmatrix) so the
// instruction stream is OMMA + trivial loads/stores -- easy to trace/simulate.
//
// Usage: omma_fp4 [iters] [warps_per_block] [blocks]
// Each warp runs `iters` loop trips of NCHAIN independent OMMAs (same operands),
// so it executes iters*NCHAIN OMMAs; the summed result is checked against
// iters*NCHAIN * (CPU reference of one MMA). Build with -DNCHAIN=1 and run
// iters=1 for the single-OMMA correctness case.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA error %s @ %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

#ifndef NCHAIN
#define NCHAIN 4  // independent accumulator chains per loop trip (hides OMMA latency)
#endif

__device__ __forceinline__
void omma(float d[4], const uint32_t a[4], const uint32_t b[2], uint32_t sfa, uint32_t sfb) {
  const uint16_t z = 0;  // byte-id / thread-id selectors for the scale registers
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
      "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}

// A: 16 rows x 32 bytes (row-major, 2 e2m1 per byte), B: 8 rows(n) x 32 bytes (k-major)
// sfa: 16 rows x 4 bytes, sfb: 8 cols x 4 bytes. D: 16x8 f32 per warp.
__global__ void omma_kernel(const uint8_t* A, const uint8_t* B, const uint8_t* sfa,
                            const uint8_t* sfb, float* D, int iters) {
  int lane = threadIdx.x & 31;
  int ai = lane & 3, aj = lane >> 2;
  uint32_t a[4], b[2];
  a[0] = *(const uint32_t*)(A + (aj    ) * 32 + ai * 4);
  a[1] = *(const uint32_t*)(A + (aj + 8) * 32 + ai * 4);
  a[2] = *(const uint32_t*)(A + (aj    ) * 32 + ai * 4 + 16);
  a[3] = *(const uint32_t*)(A + (aj + 8) * 32 + ai * 4 + 16);
  b[0] = *(const uint32_t*)(B + aj * 32 + ai * 4);
  b[1] = *(const uint32_t*)(B + aj * 32 + ai * 4 + 16);
  // scale_vec::4X, thread-id selector 0: lane 4g holds row g, lane 4g+1 row g+8 (A); lane 4g col g (B)
  uint32_t fa = *(const uint32_t*)(sfa + ((lane >> 2) + 8 * (lane & 1)) * 4);
  uint32_t fb = *(const uint32_t*)(sfb + (lane >> 2) * 4);

  float acc[NCHAIN][4] = {};
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int c = 0; c < NCHAIN; ++c) omma(acc[c], a, b, fa, fb);
  }
  float s[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    s[i] = 0.f;
#pragma unroll
    for (int c = 0; c < NCHAIN; ++c) s[i] += acc[c][i];
  }
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  float* Dw = D + warp * 128;
  int r = lane >> 2, cc = 2 * (lane & 3);
  Dw[(r    ) * 8 + cc    ] = s[0];
  Dw[(r    ) * 8 + cc + 1] = s[1];
  Dw[(r + 8) * 8 + cc    ] = s[2];
  Dw[(r + 8) * 8 + cc + 1] = s[3];
}

static float e2m1(uint8_t v) {
  static const float m[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return (v & 8 ? -1.f : 1.f) * m[v & 7];
}
static float ue4m3(uint8_t v) {  // unsigned e4m3, bias 7
  int e = (v >> 3) & 0xF, m = v & 7;
  return e == 0 ? std::ldexp(m / 8.f, -6) : std::ldexp(1.f + m / 8.f, e - 7);
}
static float fp4_at(const uint8_t* rowbytes, int k) {  // low nibble = even k
  uint8_t byte = rowbytes[k >> 1];
  return e2m1((k & 1) ? (byte >> 4) : (byte & 0xF));
}

int main(int argc, char** argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 1;
  int wpb   = argc > 2 ? atoi(argv[2]) : 1;
  int nblk  = argc > 3 ? atoi(argv[3]) : 1;
  int nwarps = wpb * nblk;

  srand(12345);
  std::vector<uint8_t> hA(16 * 32), hB(8 * 32), hSA(16 * 4), hSB(8 * 4);
  for (auto& x : hA) x = rand() & 0xFF;
  for (auto& x : hB) x = rand() & 0xFF;
  // scales in [2^-2, 2^2) with random mantissa -> non-uniform, exactly representable
  for (auto& x : hSA) x = (uint8_t)(((5 + rand() % 4) << 3) | (rand() & 7));
  for (auto& x : hSB) x = (uint8_t)(((5 + rand() % 4) << 3) | (rand() & 7));

  // CPU reference for ONE OMMA (double accumulation)
  std::vector<double> ref(128);
  for (int r = 0; r < 16; ++r)
    for (int n = 0; n < 8; ++n) {
      double s = 0;
      for (int k = 0; k < 64; ++k)
        s += (double)fp4_at(&hA[r * 32], k) * ue4m3(hSA[r * 4 + k / 16]) *
             (double)fp4_at(&hB[n * 32], k) * ue4m3(hSB[n * 4 + k / 16]);
      ref[r * 8 + n] = s;
    }

  uint8_t *dA, *dB, *dSA, *dSB; float* dD;
  CK(cudaMalloc(&dA, hA.size())); CK(cudaMalloc(&dB, hB.size()));
  CK(cudaMalloc(&dSA, hSA.size())); CK(cudaMalloc(&dSB, hSB.size()));
  CK(cudaMalloc(&dD, nwarps * 128 * sizeof(float)));
  CK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSA, hSA.data(), hSA.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSB, hSB.data(), hSB.size(), cudaMemcpyHostToDevice));

  omma_kernel<<<nblk, 32 * wpb>>>(dA, dB, dSA, dSB, dD, iters);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());

  std::vector<float> hD(nwarps * 128);
  CK(cudaMemcpy(hD.data(), dD, hD.size() * sizeof(float), cudaMemcpyDeviceToHost));
  // every OMMA adds the same product, so expected = iters*NCHAIN*ref (f32 accumulation)
  double mult = (double)iters * NCHAIN, max_rel = 0;
  int bad = 0;
  for (int w = 0; w < nwarps; ++w)
    for (int i = 0; i < 128; ++i) {
      double exp = mult * ref[i], got = hD[w * 128 + i];
      double rel = std::fabs(got - exp) / std::fmax(1.0, std::fabs(exp));
      max_rel = std::fmax(max_rel, rel);
      if (rel > 1e-5 * std::fmax(1.0, (double)iters)) ++bad;
    }
  printf("iters=%d nchain=%d warps=%d omma_per_warp=%d  max_rel_err=%.3g  D[0]=%.4f ref=%.4f  %s\n",
         iters, NCHAIN, nwarps, iters * NCHAIN, max_rel, hD[0], mult * ref[0], bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
