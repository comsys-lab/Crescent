// Print the device properties needed for the Blackwell Accel-Sim config.
#include <cstdio>
#include <cuda_runtime.h>
int main() {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  int clk = 0, memclk = 0; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
  cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, 0);
  printf("name=%s\ncc=%d.%d\nSMs=%d\nmax_sm_clock_khz=%d\nmem_clock_khz=%d\nmem_bus_width_bits=%d\n"
         "l2_bytes=%d\nregs_per_sm=%d\nregs_per_block=%d\nsmem_per_sm=%zu\nsmem_per_block_optin=%zu\n"
         "max_threads_per_sm=%d\nmax_blocks_per_sm=%d\nwarp=%d\nglobal_mem=%zu\n",
         p.name, p.major, p.minor, p.multiProcessorCount, clk, memclk, p.memoryBusWidth,
         p.l2CacheSize, p.regsPerMultiprocessor, p.regsPerBlock, p.sharedMemPerMultiprocessor,
         p.sharedMemPerBlockOptin, p.maxThreadsPerMultiProcessor, p.maxBlocksPerMultiProcessor,
         p.warpSize, p.totalGlobalMem);
}
