// Trivial non-tensor kernel: sanity check that NVBit tracing works at all on this GPU.
#include <cstdio>
__global__ void vecadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
int main() {
  int n = 256; float *a, *b, *c;
  cudaMallocManaged(&a, n * 4); cudaMallocManaged(&b, n * 4); cudaMallocManaged(&c, n * 4);
  for (int i = 0; i < n; ++i) { a[i] = i; b[i] = 2 * i; }
  vecadd<<<2, 128>>>(a, b, c, n);
  cudaError_t e = cudaDeviceSynchronize();
  int bad = 0; for (int i = 0; i < n; ++i) bad += c[i] != 3.f * i;
  printf("vecadd %s (%s)\n", bad ? "FAIL" : "PASS", cudaGetErrorString(e));
  return bad;
}
