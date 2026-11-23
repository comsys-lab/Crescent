// clock64-based OMMA microbenchmark on one SM (one block):
//   latency mode:    1 warp, 1 dependent accumulator chain  -> cycles per OMMA = latency
//   throughput mode: W warps x NCH independent chains        -> SM cycles per OMMA
// Usage: omma_latency_tput <warps> <chains: 1|4|8> [iters]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

__device__ __forceinline__
void omma(float d[4], const uint32_t a[4], const uint32_t b[2], uint32_t sfa, uint32_t sfb) {
  const uint16_t z = 0;
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
      "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}

template <int NCH>
__global__ void bench(const uint32_t* in, float* out, long long* cyc, int iters) {
  uint32_t a[4] = {in[0], in[1], in[2], in[3]}, b[2] = {in[4], in[5]}, fa = in[6], fb = in[7];
  float acc[NCH][4] = {};
  __syncthreads();
  long long t0 = clock64();
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int c = 0; c < NCH; ++c) omma(acc[c], a, b, fa, fb);
  }
  __syncthreads();
  long long t1 = clock64();
  float s = 0;
#pragma unroll
  for (int c = 0; c < NCH; ++c) s += acc[c][0] + acc[c][1] + acc[c][2] + acc[c][3];
  out[threadIdx.x] = s;
  if (threadIdx.x == 0) *cyc = t1 - t0;
}

int main(int argc, char** argv) {
  int warps = argc > 1 ? atoi(argv[1]) : 1, nch = argc > 2 ? atoi(argv[2]) : 1;
  int iters = argc > 3 ? atoi(argv[3]) : 4096;
  uint32_t h[8] = {0x11111111, 0x22222222, 0x11111111, 0x22222222, 0x11111111, 0x22222222, 0x38383838, 0x38383838};
  uint32_t* din; float* dout; long long* dc;
  cudaMalloc(&din, 32); cudaMalloc(&dout, 4096); cudaMalloc(&dc, 8);
  cudaMemcpy(din, h, 32, cudaMemcpyHostToDevice);
  auto run = [&]() {
    if (nch == 1) bench<1><<<1, 32 * warps>>>(din, dout, dc, iters);
    else if (nch == 4) bench<4><<<1, 32 * warps>>>(din, dout, dc, iters);
    else bench<8><<<1, 32 * warps>>>(din, dout, dc, iters);
  };
  run(); cudaDeviceSynchronize();  // warm-up
  run(); cudaError_t e = cudaDeviceSynchronize();
  long long c; cudaMemcpy(&c, dc, 8, cudaMemcpyDeviceToHost);
  double total = (double)warps * nch * iters;
  printf("warps=%d chains=%d iters=%d cycles=%lld  cycles/OMMA(per warp-chain)=%.2f  SM cycles/OMMA=%.3f  (%s)\n",
         warps, nch, iters, c, (double)c / iters / nch * (nch == 1 ? 1 : 1), c / total, cudaGetErrorString(e));
}
