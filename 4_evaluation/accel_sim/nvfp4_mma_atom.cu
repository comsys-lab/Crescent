// NVFP4 16x64x8 MMA. A, B, SFA, SFB, and the largest-error-selection metadata
// (metaA, metaB) are all moved global -> shared memory with TMA (cp.async.bulk.tensor,
// 6 descriptor-driven bulk copies sharing one mbarrier -- same pattern CUTLASS's
// SM120 collective uses for A/B/SFA/SFB, extended here to the metadata too).
// A/B's MMA fragments are then loaded from shared memory with ldmatrix (SASS LDSM);
// SFA/SFB/metaA/metaB are read from shared memory with a plain scattered load
// (too small to bother with ldmatrix).
// metaA/metaB are 1 byte per 8-element group: bits[2:0] = index of the largest-error
// element in the group, bits[4:3] = its promoted E2M3 mantissa correction. Not yet
// consumed by the MMA in this kernel -- moved and verified only (see README).
// Validated against a trusted reference (plain scattered ld.global, no SMEM/TMA/LDSM)
// on NON-UNIFORM input data (constant inputs can't catch a wrong fragment permutation).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

#define CU_CHECK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { \
  const char* s; cuGetErrorString(r_, &s); printf("CU error %s @ %d: %s\n", #x, __LINE__, s); std::exit(1); } } while (0)
#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA error %s @ %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

__device__ __forceinline__
void mma_m16n8k64_nvfp4(float d[4], const uint32_t a[4], const uint32_t b[2],
                        uint32_t sfa, uint32_t sfb) {
  const uint16_t z = 0;
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
      "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}
__device__ __forceinline__
void ldmatrix_x4(uint32_t d[4], const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a));
}
__device__ __forceinline__
void ldmatrix_x2(uint32_t d[2], const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(d[0]), "=r"(d[1]) : "r"(a));
}

__device__ void epilogue(int lane, float acc[4], const float* C, float alpha, float beta, float* D) {
  int r = lane >> 2, c = 2 * (lane & 3);
  D[(r    ) * 8 + c    ] = alpha * acc[0] + beta * C[(r    ) * 8 + c    ];
  D[(r    ) * 8 + c + 1] = alpha * acc[1] + beta * C[(r    ) * 8 + c + 1];
  D[(r + 8) * 8 + c    ] = alpha * acc[2] + beta * C[(r + 8) * 8 + c    ];
  D[(r + 8) * 8 + c + 1] = alpha * acc[3] + beta * C[(r + 8) * 8 + c + 1];
}

// ---- reference: fully manual, no SMEM/TMA/LDSM (trusted, matches CUTLASS MMA_Traits) ----
__global__ void ref_kernel(const uint8_t* A, const uint8_t* B, const uint8_t* sfa, const uint8_t* sfb,
                           const float* C, float alpha, float beta, float* D) {
  int lane = threadIdx.x;
  int ai = lane & 3, aj = lane >> 2;
  uint32_t a[4], b[2];
  a[0] = *(const uint32_t*)(A + (aj    ) * 32 + ai * 4);
  a[1] = *(const uint32_t*)(A + (aj + 8) * 32 + ai * 4);
  a[2] = *(const uint32_t*)(A + (aj    ) * 32 + ai * 4 + 16);
  a[3] = *(const uint32_t*)(A + (aj + 8) * 32 + ai * 4 + 16);
  int bi = lane & 3, bj = lane >> 2;
  b[0] = *(const uint32_t*)(B + bj * 32 + bi * 4);
  b[1] = *(const uint32_t*)(B + bj * 32 + bi * 4 + 16);
  uint32_t fa = *(const uint32_t*)(sfa + ((lane >> 2) + 8 * (lane & 1)) * 4);
  uint32_t fb = *(const uint32_t*)(sfb + (lane >> 2) * 4);
  float acc[4] = {0, 0, 0, 0};
  mma_m16n8k64_nvfp4(acc, a, b, fa, fb);
  epilogue(lane, acc, C, alpha, beta, D);
}

// Register-packing step for a HYPOTHETICAL new tensor-core instruction that would take
// the largest-error-selection metadata as a direct operand (alongside a[4]/b[2]) --
// that instruction doesn't exist on real hardware, so it is never issued here. This
// only measures the cost of GETTING the packed register ready: 4 metaA reads (one per
// Q/A fragment register, matching a[0..3]) and 2 metaB reads (matching b[0..1]), each
// masked to its low 5 bits (3-bit index + 2-bit E2M3 correction) and packed into one
// uint32_t (30 of 32 bits used). The exact (row,group) <-> a[i]/b[i] correspondence
// isn't reproduced -- this measures the packing COMPUTE cost (SMEM reads + shifts/ORs
// per lane), not a real correction algorithm.
__device__ __forceinline__
uint32_t pack_qk_meta(int lane, const uint8_t* sMetaA, const uint8_t* sMetaB) {
  uint32_t p = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) p |= (uint32_t)(sMetaA[(lane * 4 + i) & 127] & 0x1F) << (5 * i);
#pragma unroll
  for (int i = 0; i < 2; ++i) p |= (uint32_t)(sMetaB[(lane * 2 + i) & 63] & 0x1F) << (20 + 5 * i);
  return p;
}

// Same packing, but sMetaA/sMetaB are the REAL on-wire format: two 5-bit fields packed
// into one 16-bit word (bits [0:5)=group0, [5:10)=group1, [10:16) unused) instead of one
// byte per group -- matches what the quantize kernel / TMA actually store. Same total
// bytes as the unpacked version (word-packing group=8 doesn't shrink the footprint, see
// nvfp4_microbench's group8 word-pack test), so this isolates whether extracting two
// 5-bit fields out of a shared 16-bit word (vs. reading one already-isolated byte) adds
// cost to the register-packing step itself.
__device__ __forceinline__
uint32_t pack_qk_meta_word16(int lane, const uint16_t* wMetaA, const uint16_t* wMetaB) {
  uint32_t p = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int idx = (lane * 4 + i) & 127;
    uint16_t w = wMetaA[(idx >> 1) & 63];
    uint32_t v = (idx & 1) ? ((w >> 5) & 0x1F) : (w & 0x1F);
    p |= v << (5 * i);
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int idx = (lane * 2 + i) & 63;
    uint16_t w = wMetaB[(idx >> 1) & 31];
    uint32_t v = (idx & 1) ? ((w >> 5) & 0x1F) : (w & 0x1F);
    p |= v << (20 + 5 * i);
  }
  return p;
}

// Linear config (NOT QK^T): group=4, metadata on the A operand (activation) only --
// the weight (B) is quantized offline with no selection, so there's nothing to pack
// for b[0..1]. Each value is 4 bits (2-bit index + 2-bit E2M3 correction, since GS=4
// needs ceil(log2(4))=2 index bits, vs QK^T's GS=8 needing 3). 4 values, matching
// a[0..3], packed into the low 16 of 32 bits -- half the width pack_qk_meta uses,
// since there's no B-side contribution here.
//
// sMetaA is nibble-packed (2 values/byte, 0 padding waste -- this is the real on-wire
// format established for group=4 in ../execution_time/: 4-bit values pack perfectly
// into a byte 2-at-a-time, unlike QK^T's 5-bit values, which don't divide evenly into
// a byte or word -- see ../execution_time/README.md "packing experiments" discussion
// if that file still documents it, or nvfp4_microbench.cu's NVFP4_NIBBLE_PACK).
__device__ __forceinline__
uint32_t pack_linear_meta(int lane, const uint8_t* sMetaA_nibble) {
  uint32_t p = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int idx = (lane * 4 + i) & 127;             // same 4-values-per-lane indexing as pack_qk_meta
    uint8_t byte = sMetaA_nibble[(idx >> 1) & 63];
    uint32_t v = (idx & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
    p |= v << (4 * i);
  }
  return p;                                      // bits [0:16) used, [16:32) unused
}

// ---- TMA (global->SMEM) + ldmatrix (SMEM->fragment) + MMA ----
// TMA moves all six operand-adjacent tensors -- A, B, SFA, SFB, metaA, metaB --
// exactly like CUTLASS's SM120 collective: separate cp.async.bulk.tensor issues,
// all sharing one mbarrier. metaA/metaB are read back here for verification (see
// metaA_out/metaB_out); when PACK_META, they're additionally bit-packed into one
// register per the hypothetical-new-instruction scheme above (see pack_qk_meta) --
// this kernel still doesn't apply any E2M3 correction to the MMA result itself.
// PACK_MODE: 0 = no packing, 1 = pack from byte-indexed sMeta (old test, doesn't match
// real storage), 2 = pack from 16-bit-word-packed sMeta (matches the real quantize-kernel
// output format: two 5-bit fields per word).
template <int PACK_MODE>
__global__ void tma_kernel(const __grid_constant__ CUtensorMap descA,
                           const __grid_constant__ CUtensorMap descB,
                           const __grid_constant__ CUtensorMap descSFA,
                           const __grid_constant__ CUtensorMap descSFB,
                           const __grid_constant__ CUtensorMap descMetaA,
                           const __grid_constant__ CUtensorMap descMetaB,
                           const float* C, float alpha, float beta, float* D,
                           uint8_t* metaA_out, uint8_t* metaB_out,
                           uint32_t* packed_out) {
  __shared__ alignas(128) uint8_t sA[16 * 32];
  __shared__ alignas(128) uint8_t sB[8 * 32];
  __shared__ alignas(128) uint8_t sSFA[16 * 4];
  __shared__ alignas(128) uint8_t sSFB[8 * 4];
  __shared__ alignas(128) uint8_t sMetaA[16 * 8];   // 16 rows x (64/8) groups/row
  __shared__ alignas(128) uint8_t sMetaB[8 * 8];    // 8 rows x (64/8) groups/row
  __shared__ alignas(8) uint64_t mbar;
  int lane = threadIdx.x;

  if (lane == 0) {
    uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(&mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(mbar_addr));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(mbar_addr),
                    "r"((uint32_t)(16 * 32 + 8 * 32 + 16 * 4 + 8 * 4 + 16 * 8 + 8 * 8)));
    uint32_t sA_addr     = (uint32_t)__cvta_generic_to_shared(sA);
    uint32_t sB_addr     = (uint32_t)__cvta_generic_to_shared(sB);
    uint32_t sSFA_addr   = (uint32_t)__cvta_generic_to_shared(sSFA);
    uint32_t sSFB_addr   = (uint32_t)__cvta_generic_to_shared(sSFB);
    uint32_t sMetaA_addr = (uint32_t)__cvta_generic_to_shared(sMetaA);
    uint32_t sMetaB_addr = (uint32_t)__cvta_generic_to_shared(sMetaB);
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];\n"
      :: "r"(sA_addr), "l"(&descA), "r"(0), "r"(0), "r"(mbar_addr));
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];\n"
      :: "r"(sB_addr), "l"(&descB), "r"(0), "r"(0), "r"(mbar_addr));
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];\n"
      :: "r"(sSFA_addr), "l"(&descSFA), "r"(0), "r"(0), "r"(mbar_addr));
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];\n"
      :: "r"(sSFB_addr), "l"(&descSFB), "r"(0), "r"(0), "r"(mbar_addr));
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];\n"
      :: "r"(sMetaA_addr), "l"(&descMetaA), "r"(0), "r"(0), "r"(mbar_addr));
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];\n"
      :: "r"(sMetaB_addr), "l"(&descMetaB), "r"(0), "r"(0), "r"(mbar_addr));
  }
  __syncthreads();
  if (lane == 0) {
    uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(&mbar);
    int done = 0;
    while (!done) {
      asm volatile(
        "{\n.reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\n"
        "selp.b32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(mbar_addr));
    }
  }
  __syncthreads();   // all lanes wait until lane 0 has observed TMA completion

  // A fragment: ldmatrix.x4 -> groups of 8 lanes each supply one 8-row tile's row address
  int g = lane >> 3, rig = lane & 7;
  int arow = rig + 8 * (g & 1);
  int aoff = 16 * (g >> 1);
  uint32_t a[4];
  ldmatrix_x4(a, sA + arow * 32 + aoff);

  // B fragment: ldmatrix.x2 -> B only has 8 rows
  int brow = lane & 7;
  int boff = 16 * ((lane >> 3) & 1);
  uint32_t b[2];
  ldmatrix_x2(b, sB + brow * 32 + boff);

  uint32_t fa = *(const uint32_t*)(sSFA + ((lane >> 2) + 8 * (lane & 1)) * 4);
  uint32_t fb = *(const uint32_t*)(sSFB + (lane >> 2) * 4);
  float acc[4] = {0, 0, 0, 0};
  mma_m16n8k64_nvfp4(acc, a, b, fa, fb);
  epilogue(lane, acc, C, alpha, beta, D);

  // metaA/metaB: not consumed by the MMA yet -- read back out to global so
  // main() can verify the TMA move landed correctly (byte-for-byte vs source).
  for (int i = lane; i < 16 * 8; i += 32) metaA_out[i] = sMetaA[i];
  for (int i = lane; i < 8 * 8; i += 32)  metaB_out[i] = sMetaB[i];

  if (PACK_MODE == 1) {
    packed_out[lane] = pack_qk_meta(lane, sMetaA, sMetaB);
  } else if (PACK_MODE == 2) {
    packed_out[lane] = pack_qk_meta_word16(lane, (const uint16_t*)sMetaA, (const uint16_t*)sMetaB);
  }
}

static CUtensorMap make_desc_2d(void* base, uint64_t rowBytes, uint64_t rows) {
  CUtensorMap desc{};
  cuuint64_t globalDim[2]     = {rowBytes, rows};
  cuuint64_t globalStrides[1] = {rowBytes};
  cuuint32_t boxDim[2]        = {(cuuint32_t)rowBytes, (cuuint32_t)rows};
  cuuint32_t elementStrides[2] = {1, 1};
  CU_CHECK(cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base,
      globalDim, globalStrides, boxDim, elementStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return desc;
}


int main() {
  CU_CHECK(cuInit(0));
  CUdevice dev; CU_CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx; CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CU_CHECK(cuCtxSetCurrent(ctx));

  uint8_t *A, *B, *sfa, *sfb, *metaA, *metaB, *metaA_out, *metaB_out;
  float *C, *Dref, *Dtma;
  uint32_t *packed_out;
  CK(cudaMallocManaged(&A, 16 * 32)); CK(cudaMallocManaged(&B, 8 * 32));
  CK(cudaMallocManaged(&sfa, 16 * 4)); CK(cudaMallocManaged(&sfb, 8 * 4));
  CK(cudaMallocManaged(&metaA, 16 * 8)); CK(cudaMallocManaged(&metaB, 8 * 8));
  CK(cudaMallocManaged(&metaA_out, 16 * 8)); CK(cudaMallocManaged(&metaB_out, 8 * 8));
  CK(cudaMallocManaged(&C, 16 * 8 * sizeof(float)));
  CK(cudaMallocManaged(&Dref, 16 * 8 * sizeof(float)));
  CK(cudaMallocManaged(&Dtma, 16 * 8 * sizeof(float)));
  CK(cudaMallocManaged(&packed_out, 32 * sizeof(uint32_t)));

  // non-uniform test data: a wrong fragment permutation would change the result
  // (constant inputs can't distinguish a wrong permutation from a right one).
  for (int i = 0; i < 16 * 32; i++) A[i] = (uint8_t)(0x11 + (i % 7) * 0x11);
  for (int i = 0; i < 8 * 32; i++)  B[i] = (uint8_t)(0x22 + (i % 5) * 0x11);
  for (int i = 0; i < 16 * 4; i++)  sfa[i] = (uint8_t)(0x38 + (i % 3));
  for (int i = 0; i < 8 * 4; i++)   sfb[i] = (uint8_t)(0x38 + (i % 4));
  for (int i = 0; i < 16 * 8; i++)  C[i] = (float)(i % 13) * 0.5f;
  // metaA/metaB: 1 byte/8-elem group, bits[2:0]=index, bits[4:3]=E2M3 correction
  for (int i = 0; i < 16 * 8; i++) metaA[i] = (uint8_t)(((i % 4) << 3) | (i % 7));
  for (int i = 0; i < 8 * 8; i++)  metaB[i] = (uint8_t)(((i % 3) << 3) | (i % 5));

  CUtensorMap descA     = make_desc_2d(A,   32, 16);
  CUtensorMap descB     = make_desc_2d(B,   32, 8);
  CUtensorMap descSFA   = make_desc_2d(sfa, 16 * 4, 1);   // whole 64B as one "row" (multiple of 16B)
  CUtensorMap descSFB   = make_desc_2d(sfb, 8 * 4, 1);     // whole 32B as one "row"
  CUtensorMap descMetaA = make_desc_2d(metaA, 16 * 8, 1);  // whole 128B as one "row"
  CUtensorMap descMetaB = make_desc_2d(metaB, 8 * 8, 1);   // whole 64B as one "row"

  ref_kernel<<<1, 32>>>(A, B, sfa, sfb, C, 1.3f, 0.7f, Dref);
  tma_kernel<0><<<1, 32>>>(descA, descB, descSFA, descSFB, descMetaA, descMetaB,
                               C, 1.3f, 0.7f, Dtma, metaA_out, metaB_out, packed_out);
  CK(cudaDeviceSynchronize());

  int mismatches = 0;
  for (int i = 0; i < 16 * 8; i++) if (Dref[i] != Dtma[i]) mismatches++;
  int meta_mismatches = 0;
  for (int i = 0; i < 16 * 8; i++) if (metaA[i] != metaA_out[i]) meta_mismatches++;
  for (int i = 0; i < 8 * 8; i++)  if (metaB[i] != metaB_out[i]) meta_mismatches++;

  printf("ref D[0:8] :"); for (int i = 0; i < 8; i++) printf(" %8.3f", Dref[i]); printf("\n");
  printf("tma D[0:8] :"); for (int i = 0; i < 8; i++) printf(" %8.3f", Dtma[i]); printf("\n");
  printf("mismatches: %d / %d   [%s]\n", mismatches, 16 * 8, mismatches == 0 ? "MATCH" : "MISMATCH");
  printf("meta mismatches: %d / %d   [%s]\n", meta_mismatches, 16 * 8 + 8 * 8,
         meta_mismatches == 0 ? "MATCH" : "MISMATCH");
  if (mismatches != 0 || meta_mismatches != 0) return 1;

  // ---- timing: cost of register-packing the metadata for a hypothetical new tensor-core
  // instruction, vs not doing it. That instruction is never issued (doesn't exist on real
  // hardware) -- this only measures getting the packed register ready.
  const int warmup = 2000, iters = 20000;
  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));

  for (int i = 0; i < warmup; ++i)
    tma_kernel<0><<<1, 32>>>(descA, descB, descSFA, descSFB, descMetaA, descMetaB,
                                 C, 1.3f, 0.7f, Dtma, metaA_out, metaB_out, packed_out);
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(t0));
  for (int i = 0; i < iters; ++i)
    tma_kernel<0><<<1, 32>>>(descA, descB, descSFA, descSFB, descMetaA, descMetaB,
                                 C, 1.3f, 0.7f, Dtma, metaA_out, metaB_out, packed_out);
  CK(cudaEventRecord(t1));
  CK(cudaEventSynchronize(t1));
  float ms_nopack = 0.f; CK(cudaEventElapsedTime(&ms_nopack, t0, t1));
  ms_nopack /= iters;

  for (int i = 0; i < warmup; ++i)
    tma_kernel<1><<<1, 32>>>(descA, descB, descSFA, descSFB, descMetaA, descMetaB,
                                C, 1.3f, 0.7f, Dtma, metaA_out, metaB_out, packed_out);
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(t0));
  for (int i = 0; i < iters; ++i)
    tma_kernel<1><<<1, 32>>>(descA, descB, descSFA, descSFB, descMetaA, descMetaB,
                                C, 1.3f, 0.7f, Dtma, metaA_out, metaB_out, packed_out);
  CK(cudaEventRecord(t1));
  CK(cudaEventSynchronize(t1));
  float ms_pack = 0.f; CK(cudaEventElapsedTime(&ms_pack, t0, t1));
  ms_pack /= iters;

  for (int i = 0; i < warmup; ++i)
    tma_kernel<2><<<1, 32>>>(descA, descB, descSFA, descSFB, descMetaA, descMetaB,
                             C, 1.3f, 0.7f, Dtma, metaA_out, metaB_out, packed_out);
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(t0));
  for (int i = 0; i < iters; ++i)
    tma_kernel<2><<<1, 32>>>(descA, descB, descSFA, descSFB, descMetaA, descMetaB,
                             C, 1.3f, 0.7f, Dtma, metaA_out, metaB_out, packed_out);
  CK(cudaEventRecord(t1));
  CK(cudaEventSynchronize(t1));
  float ms_pack_word16 = 0.f; CK(cudaEventElapsedTime(&ms_pack_word16, t0, t1));
  ms_pack_word16 /= iters;

  printf("\nno-pack          : %.6f ms/iter\n", ms_nopack);
  printf("pack (byte-idx)  : %.6f ms/iter\n", ms_pack);
  printf("pack (word16, real fmt): %.6f ms/iter  (%+.2f%% vs no-pack)\n",
         ms_pack_word16, 100.0 * (ms_pack_word16 - ms_nopack) / ms_nopack);
  printf("packing adds %.6f ms/iter (%+.2f%%)\n",
         ms_pack - ms_nopack, 100.0 * (ms_pack - ms_nopack) / ms_nopack);

  return 0;
}
