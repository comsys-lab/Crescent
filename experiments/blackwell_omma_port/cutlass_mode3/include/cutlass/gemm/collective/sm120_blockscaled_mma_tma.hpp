/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/trace.h"
#include "cutlass/numeric_types.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/functional.hpp"
#include <cuda_fp4.h>
#include <cuda_fp16.h>
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

// Largest-error-selection metadata config for this collective -- toggled via -D at
// build time (see nvfp4_microbench/code/build.sh), two named scenarios:
//   Linear (A@W^T): NVFP4_META_GROUP_SIZE=4, NVFP4_META_ON_B=0 -- only the activation
//     (A) is quantized per-forward and gets selection; the weight (B) is quantized
//     offline, no metadata for it.
//   QK^T (attention scores): NVFP4_META_GROUP_SIZE=8, NVFP4_META_ON_B=1 -- both Q (A)
//     and K (B) are activations computed per-forward, so both get selection metadata;
//     within each 16-element (head_dim chunk) dot product, 2 elements are picked from
//     Q's 16 and 2 from K's 16 (1 per 8-group, matching group size 8 independently on
//     each operand).
#ifndef NVFP4_META_GROUP_SIZE
#define NVFP4_META_GROUP_SIZE 4
#endif
#ifndef NVFP4_META_ON_B
#define NVFP4_META_ON_B 0
#endif
// NVFP4_META_SW_CORRECT: how the 2 extra mantissa bits are applied (see mma() below).
//   0 = not applied: metadata is only TMA-loaded (the cost a hardware implementation would
//       pay), the MMA result is plain NVFP4.
//   1 = software, CUDA cores: FMAs into the MMA accumulator.
//   2 = software, tensor cores: the residual R (exactly E2M1 x sfa/4) is built in registers
//       from the metadata and fed to extra FP4 MMAs (Linear: A4*B + R_A*B; QK^T: 4 MMAs).
#ifndef NVFP4_META_SW_CORRECT
#define NVFP4_META_SW_CORRECT 0
#endif
// NVFP4_META_DCODE=1 (tensor-core mode only): meta byte carries the residual's E2M1 code directly.
#ifndef NVFP4_META_DCODE
#define NVFP4_META_DCODE 0
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// NVFP4_META_SW_CORRECT == 3: pack-only, no correction applied. Builds the SAME
// packed register pack_qk_meta()/pack_linear_meta() (nvfp4_mma_atom.cu) compute in
// isolation -- 4 values for A (standing in for a[0..3]; +2 for B matching b[0..1] when
// NVFP4_META_ON_B) -- but does it with REAL SMEM reads inside the REAL, full-pipeline
// CUTLASS mainloop instead of a standalone toy kernel, so the cost is measurable on
// real hardware at realistic occupancy/scheduling. The packed value is stored into
// g_nvfp4_pack_sink (not fed to any MMA -- no real instruction has a 13th operand)
// purely to stop the compiler from eliminating the otherwise-unused computation as
// dead code. One slot per CTA, indexed by blockIdx.x -- a PLAIN store (not atomic):
// an earlier version used atomicXor into a single shared slot, which measured a
// ~50% GEMM-time increase that turned out to be almost entirely artificial --
// -Xptxas -v showed IDENTICAL register/smem usage with vs without packing (168 regs,
// 0 spills, both configs), ruling out occupancy/register pressure, which pointed at
// the atomic itself: every CTA on the GPU, every k_block, hammering one global
// address serializes across the whole device. Indexing by blockIdx.x (distinct slot
// per CTA, so no cross-CTA contention) and dropping the atomic (a plain store to a
// runtime-dependent global address is still a visible side effect the compiler can't
// prove dead, so DCE-prevention doesn't need atomicity) removes that artifact. See
// mma() below (NVFP4_META_SW_CORRECT == 3 branch) for the approximation this makes
// (real accumulator-row indexing stands in for operand-register indexing -- right
// SMEM-access volume/pattern, not a claimed exact correspondence to any specific
// future instruction's a[i]/b[i]).
//
// Operand-count design (confirmed against real SASS): one OMMA always reads exactly
// ONE extra register for A's metadata -- 12 -> 13 operands, same as Linear. But the
// CONTENT of that register is per-M-atom (SFA already uses a different register per
// M atom in real SASS, for the same reason: A data differs by M atom), so a warp
// must build MMA_M distinct packed values per k_block, not one -- each of the warp's
// OMMAs picks whichever matches its own M atom. QK^T additionally has B's metadata,
// which is per-N-atom (MMA_N distinct values) -- modeled as a SEPARATE 14th operand,
// not merged into A's 13th, because merging would force MMA_M*MMA_N distinct
// combinations per k_block (one per (M atom, N atom) pair) instead of MMA_M + MMA_N
// (mirrors why real hardware already keeps SFA/SFB separate rather than combined).
#if NVFP4_META_SW_CORRECT == 3
constexpr int NVFP4_PACK_SINK_SLOTS = 2048;   // >= max CTAs/wave on any SM120 part
__device__ uint32_t g_nvfp4_pack_sink[NVFP4_PACK_SINK_SLOTS];
#endif

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  int SchedulerPipelineStageCount,
  class ClusterShape,
  class KernelScheduleType,
  class TileShape_,
  class ElementPairA_,
  class StridePairA_,
  class ElementPairB_,
  class StridePairB_,
  class TiledMma_,
  class GmemTiledCopyPairA_,
  class SmemLayoutAtomsA_,
  class SmemCopyAtomsA_,
  class TransformA_,
  class GmemTiledCopyPairB_,
  class SmemLayoutAtomsB_,
  class SmemCopyAtomsB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm120TmaWarpSpecializedBlockScaled<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>,
    TileShape_,
    ElementPairA_,
    StridePairA_,
    ElementPairB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyPairA_,
    SmemLayoutAtomsA_,
    SmemCopyAtomsA_,
    TransformA_,
    GmemTiledCopyPairB_,
    SmemLayoutAtomsB_,
    SmemCopyAtomsB_,
    TransformB_> {
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm120TmaWarpSpecializedBlockScaled<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>;
  using TileShape = TileShape_;
  using ElementPairA = ElementPairA_;
  using ElementPairB = ElementPairB_;
  using StridePairA = StridePairA_;
  using StridePairB = StridePairB_;

  static_assert(cute::is_same_v<remove_cvref_t<decltype(get<1>(ElementPairA{}))>,
                                remove_cvref_t<decltype(get<1>(ElementPairB{}))>>, "SFA and SFB data types should be the same");

  using RuntimeDataTypeA = void*;
  using RuntimeDataTypeB = void*;

   // A and B matrices
  using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;
  using StrideA  = remove_cvref_t<decltype(get<0>(StridePairA{}))>;

  using ElementB = remove_cvref_t<decltype(get<0>(ElementPairB{}))>;
  using StrideB  = remove_cvref_t<decltype(get<0>(StridePairB{}))>;

  // SFA and SFB
  using ElementSF = remove_cvref_t<decltype(get<1>(ElementPairA{}))>;
  using LayoutSFA = remove_cvref_t<decltype(get<1>(StridePairA{}))>;
  using LayoutSFB = remove_cvref_t<decltype(get<1>(StridePairB{}))>;

  // metaA (and metaB when NVFP4_META_ON_B): largest-error-selection metadata, 1 byte
  // per NVFP4_META_GROUP_SIZE-element K-group. Added directly to this specialization
  // (not threaded through the CollectiveBuilder's ElementPair/StridePair machinery
  // like SFA/SFB) -- plain row-major (M or N, K/GROUP_SIZE, L) uint8 array(s), same
  // stride *type* as A/B since it's the same row-major shape family.
  using ElementMeta = uint8_t;
#ifndef NVFP4_META_TRANSPOSE
#define NVFP4_META_TRANSPOSE 0
#endif
// NVFP4_META_PAIR16=1 (GS=16 only): row-pair layout. Rows r and r+8 of each 16-row
// block share one physical row of 2*K/16 = K/8 bytes (16 B at K=128, which satisfies
// TMA's 16-B stride rule that a plain K/16 = 8-B row breaks), ordered so the four
// groups a thread's A fragment a[0..3] needs are 4 consecutive bytes:
//   physical row  p   = (r / 16) * 8 + r % 8
//   byte in row       = (g / 4) * 8 + (g % 2) * 4 + ((g % 4) / 2) * 2 + (r / 8) % 2
// (g = 16-element group index along K). The GMEM/SMEM tensors are (R/2, K/8, L).
#ifndef NVFP4_META_PAIR16
#define NVFP4_META_PAIR16 0
#endif
// NVFP4_META_PREPACK=1 (GS=8, Linear): correction data stored by the quantizer already in the
// 13th-operand format -- for OMMA m16n8k64, thread ai = lane%4 holds rows r, r+8 (r%16 < 8) and
// 8-element groups ai, 4+ai of each 64-K k_block; one 32-bit word per (row pair, k_block, ai),
// one 5-bit value per byte: [c(r,ai), c(r,4+ai), c(r+8,ai), c(r+8,4+ai)], at physical row
// (r/16)*8 + r%8, byte (k/64)*16 + ai*4 of an (M/2, K/4)-byte row-major tensor (same size as
// the plain (M, K/8) layout). The GEMM reads its operand with one 32-bit LDS, no bit assembly.
#ifndef NVFP4_META_PREPACK
#define NVFP4_META_PREPACK 0
#endif
#if NVFP4_META_PAIR16
  static_assert(NVFP4_META_GROUP_SIZE == 16 && !NVFP4_META_TRANSPOSE, "PAIR16 needs GS=16, row-major");
#define NVFP4_META_RDIV 2
#define NVFP4_META_CDIV 8
#elif NVFP4_META_PREPACK
  static_assert(NVFP4_META_GROUP_SIZE == 8 && !NVFP4_META_TRANSPOSE && !NVFP4_META_ON_B, "PREPACK: Linear GS=8, row-major");
#define NVFP4_META_RDIV 2
#define NVFP4_META_CDIV 4
#else
#define NVFP4_META_RDIV 1
#define NVFP4_META_CDIV NVFP4_META_GROUP_SIZE
#endif
#if NVFP4_META_TRANSPOSE
  // Experiment: store metaA/metaB TRANSPOSED -- (M, K/GS, L) with M contiguous (stride 1)
  // instead of row-major K/GS-contiguous. Makes each TMA "row" M bytes wide (well past the
  // 16B floor) instead of K/GS bytes wide, at the cost of a strided (non-coalesced) write
  // in the quantize kernel. Tests whether a wider-but-fewer-rows TMA box shape changes the
  // GEMM-side load cost for this already-tiny tensor.
  using LayoutMetaA = cute::Stride<cute::Int<1>, int64_t, int64_t>;
#if NVFP4_META_ON_B
  using LayoutMetaB = cute::Stride<cute::Int<1>, int64_t, int64_t>;
#endif
#else
  using LayoutMetaA = StrideA;
#if NVFP4_META_ON_B
  using LayoutMetaB = StrideB;
#endif
#endif

  using ArrayElementA = ElementA;
  using ArrayElementB = ElementB;

  using TiledMma = TiledMma_;
  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using ElementAccumulator = typename TiledMma::ValTypeC;

  static constexpr int SFVecSize = TiledMma::Traits::SFVecSize;
  using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;

  // Gmem copies
  using GmemTiledCopyPairA = GmemTiledCopyPairA_;
  using GmemTiledCopyPairB = GmemTiledCopyPairB_;
  using GmemTiledCopyA    = remove_cvref_t<decltype(get<0>(GmemTiledCopyPairA{}))>;
  using GmemTiledCopySFA  = remove_cvref_t<decltype(get<1>(GmemTiledCopyPairA{}))>;
  using GmemTiledCopyB    = remove_cvref_t<decltype(get<0>(GmemTiledCopyPairB{}))>;
  using GmemTiledCopySFB  = remove_cvref_t<decltype(get<1>(GmemTiledCopyPairB{}))>;
  // metaA/metaB TMA copy -- fixed (not threaded through GmemTiledCopyPair), same
  // SM90_TMA_LOAD atom as everything else in this collective.
  using GmemTiledCopyMetaA = SM90_TMA_LOAD;
#if NVFP4_META_ON_B
  using GmemTiledCopyMetaB = SM90_TMA_LOAD;
#endif

  // Smem copies
  using SmemLayoutAtomsA = SmemLayoutAtomsA_;
  using SmemLayoutAtomsB = SmemLayoutAtomsB_;

  using SmemLayoutAtomA   = remove_cvref_t<decltype(get<0>(SmemLayoutAtomsA{}))>;
  using SmemLayoutAtomSFA = remove_cvref_t<decltype(get<1>(SmemLayoutAtomsA{}))>;
  using SmemLayoutAtomB   = remove_cvref_t<decltype(get<0>(SmemLayoutAtomsB{}))>;
  using SmemLayoutAtomSFB = remove_cvref_t<decltype(get<1>(SmemLayoutAtomsB{}))>;

  using SmemCopyAtomsA =  SmemCopyAtomsA_;
  using SmemCopyAtomsB =  SmemCopyAtomsB_;

  using SmemCopyAtomA   = remove_cvref_t<decltype(get<0>(SmemCopyAtomsA{}))>;
  using SmemCopyAtomSFA = remove_cvref_t<decltype(get<1>(SmemCopyAtomsA{}))>;

  using SmemCopyAtomB   = remove_cvref_t<decltype(get<0>(SmemCopyAtomsB{}))>;
  using SmemCopyAtomSFB = remove_cvref_t<decltype(get<1>(SmemCopyAtomsB{}))>;

  using TransformA = TransformA_;
  using TransformB = TransformB_;

  using ArchTag = typename DispatchPolicy::ArchTag;

  static constexpr int ThreadCount = size(TiledMma{});

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState  = typename cutlass::PipelineState<DispatchPolicy::Stages>;

  // One threads per CTA are producers (1 for operand tile)
  static constexpr int NumProducerThreadEvents = 1;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(not cute::is_void_v<SmemCopyAtomA>,
    "SM120 mainloop must specify a copy atom for A operand smem->rmem reads.");
  static_assert(not cute::is_void_v<SmemCopyAtomB>,
    "SM120 mainloop must specify a copy atom for B operand smem->rmem reads.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  // SmemLayoutAtomSFA and SmemLayoutAtomSFB are for whole CTA tiles. We add the number of pipeline stages here.
  // The number of pipeline stages is the same as the number of pipeline stages from AB Load <-> MainLoop
  using SmemLayoutSFA = decltype(make_layout(
    append(shape(SmemLayoutAtomSFA{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))
  ));

  using SmemLayoutSFB_ = decltype(make_layout(
    append(shape(SmemLayoutAtomSFB{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))
  ));

  using TileShapeSFB = cute::conditional_t<size<1>(TileShape{}) < 128,
    decltype(cute::make_shape(
      shape<0>(TileShape{}),
      Int<128>{},
      shape<2>(TileShape{}))),
    TileShape>;

  using SmemLayoutSFB = cute::conditional_t<size<1>(TileShape{}) < 128,
    decltype(cute::logical_divide(SmemLayoutSFB_{}, select<1,2>(TileShape{}))),
    SmemLayoutSFB_>;

  // metaA/metaB SMEM layout: plain row-major (CtaM/CtaN, CtaK/GROUP_SIZE) per stage, no
  // swizzle, no SF-style 128-wide broadcast (metadata is genuinely per-row, unlike the
  // scale factor's block-shared layout) -- mirrors SmemLayoutSFA's "append Stages" pattern.
  // Under NVFP4_META_TRANSPOSE, the atom is (CtaK/GROUP_SIZE, CtaM) with CtaM contiguous,
  // matching the transposed GMEM layout -- a wide (CtaM-byte) row instead of a narrow
  // (CtaK/GROUP_SIZE-byte) one.
#if NVFP4_META_TRANSPOSE
  // Same (CtaM, CtaK/GS) shape/order as the non-transposed atom (so the box shape passed
  // to make_tma_copy and tensor_metaa's own (M,K/GS,L) shape don't need to change) -- only
  // the STRIDE flips, making CtaM (not CtaK/GS) the contiguous dimension.
  using SmemLayoutAtomMetaA = decltype(make_layout(
    make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}) / Int<NVFP4_META_GROUP_SIZE>{}),
    make_stride(Int<1>{}, shape<0>(TileShape{}))));
#if NVFP4_META_ON_B
  using SmemLayoutAtomMetaB = decltype(make_layout(
    make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}) / Int<NVFP4_META_GROUP_SIZE>{}),
    make_stride(Int<1>{}, shape<1>(TileShape{}))));
#endif
#else
  using SmemLayoutAtomMetaA = decltype(make_layout(
    make_shape(shape<0>(TileShape{}) / Int<NVFP4_META_RDIV>{}, shape<2>(TileShape{}) / Int<NVFP4_META_CDIV>{}),
    make_stride(shape<2>(TileShape{}) / Int<NVFP4_META_CDIV>{}, Int<1>{})));
#if NVFP4_META_ON_B
  using SmemLayoutAtomMetaB = decltype(make_layout(
    make_shape(shape<1>(TileShape{}) / Int<NVFP4_META_RDIV>{}, shape<2>(TileShape{}) / Int<NVFP4_META_CDIV>{}),
    make_stride(shape<2>(TileShape{}) / Int<NVFP4_META_CDIV>{}, Int<1>{})));
#endif
#endif

  // Fully pipelined, same as A/B/SFA/SFB -- properly double/multi-buffered across
  // Stages so genuine pipeline overlap is preserved. This needs extra SMEM headroom
  // (StageCountAutoCarveout picks Stages from the ORIGINAL per-stage size and has no
  // way to know about this addition), freed up on the caller side via an inflated
  // epilogue carveout that drops Stages by exactly one -- see nvfp4_microbench.cu.
  using SmemLayoutMetaA = decltype(make_layout(
    append(shape(SmemLayoutAtomMetaA{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomMetaA{}), size(SmemLayoutAtomMetaA{}))));
#if NVFP4_META_ON_B
  using SmemLayoutMetaB = decltype(make_layout(
    append(shape(SmemLayoutAtomMetaB{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomMetaB{}), size(SmemLayoutAtomMetaB{}))));
#endif

  static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
  static_assert(rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operands from rmem for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD>, "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD>, "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  static constexpr bool IsF8F6F4 = detail::is_sm120_f8f6f4<TiledMma, ElementA, ElementB>();

  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  using TmaInternalElementA = cute::conditional_t<not IsF8F6F4,
                                                  ElementA,
                              cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e2m1_t>,
                                                  cutlass::detail::float_e2m1_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e2m3_t>,
                                                cutlass::detail::float_e2m3_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e3m2_t>,
                                                cutlass::detail::float_e3m2_unpacksmem_t,
                                                uint_bit_t<sizeof_bits_v<ElementA>>>>>>;

  using TmaInternalElementB = cute::conditional_t<not IsF8F6F4,
                                                  ElementB,
                              cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e2m1_t>,
                                                  cutlass::detail::float_e2m1_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e2m3_t>,
                                                cutlass::detail::float_e2m3_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e3m2_t>,
                                                cutlass::detail::float_e3m2_unpacksmem_t,
                                                uint_bit_t<sizeof_bits_v<ElementB>>>>>>;

  using TmaInternalElementSF = ElementSF;

  using SmemAllocTypeA = cute::conditional_t<IsF8F6F4, uint8_t, typename TiledMma::ValTypeA>;
  using SmemAllocTypeB = cute::conditional_t<IsF8F6F4, uint8_t, typename TiledMma::ValTypeB>;

  // Set the bytes transferred in this TMA transaction (may involve multiple issues)
  static constexpr uint32_t TmaTransactionBytesMK = static_cast<uint32_t>(
    cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSF>) +
    cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutA{})) * sizeof_bits<ElementA>::value) +
    cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutMetaA{})) * cute::sizeof_bits_v<ElementMeta>));

#if NVFP4_META_ON_B
  static constexpr uint32_t TmaTransactionBytesNK = static_cast<uint32_t>(
    cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSF>) +
    cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutB{})) * sizeof_bits<ElementB>::value) +
    cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutMetaB{})) * cute::sizeof_bits_v<ElementMeta>));
#else
  // B (weight) is quantized offline -- no selection metadata, no metaB TMA copy.
  static constexpr uint32_t TmaTransactionBytesNK = static_cast<uint32_t>(
    cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSF>) +
    cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutB{})) * sizeof_bits<ElementB>::value));
#endif

  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK;

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      alignas(1024) cute::ArrayEngine<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      alignas(1024) cute::ArrayEngine<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
      alignas(16)   cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
      alignas(16)   cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
      alignas(16)   cute::ArrayEngine<ElementMeta, cute::cosize_v<SmemLayoutMetaA>> smem_MetaA;
#if NVFP4_META_ON_B
      alignas(16)   cute::ArrayEngine<ElementMeta, cute::cosize_v<SmemLayoutMetaB>> smem_MetaB;
#endif
    } tensors;
    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    alignas(16) PipelineStorage pipeline_storage;
  };

  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A{nullptr};
    StrideA dA{};
    ElementB const* ptr_B{nullptr};
    StrideB dB{};
    ElementSF const* ptr_SFA{nullptr};
    LayoutSFA layout_SFA{};
    ElementSF const* ptr_SFB{nullptr};
    LayoutSFB layout_SFB{};
    ElementMeta const* ptr_MetaA{nullptr};
    LayoutMetaA layout_MetaA{};
#if NVFP4_META_ON_B
    ElementMeta const* ptr_MetaB{nullptr};
    LayoutMetaB layout_MetaB{};
#endif
  };

  // Device side kernel params
  struct Params {
    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A = decltype(make_tma_copy(
        GmemTiledCopyA{},
        make_tensor(recast_ptr<TmaInternalElementA>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{}));  // No programmatic multicast
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy(
        GmemTiledCopyB{},
        make_tensor(recast_ptr<TmaInternalElementB>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        _1{}));  // No programmatic multicast

    using TMA_SFA = decltype(make_tma_copy<uint16_t>(
        GmemTiledCopySFA{},
        make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSFA{}),
        SmemLayoutSFA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{}));  // No programmatic multicast


    using TMA_SFB = decltype(make_tma_copy<uint16_t>(
        GmemTiledCopySFB{},
        make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSFB{}),
        SmemLayoutSFB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShapeSFB{}), shape<2>(TileShapeSFB{})),
        _1{}));  // No programmatic multicast

    // Selection group size = NVFP4_META_GROUP_SIZE elements (2-bit index + 2-bit E2M3
    // correction when GROUP_SIZE==4, 3-bit index + 2-bit correction when GROUP_SIZE==8;
    // stored unpacked as 1 byte/group either way).
    using TMA_MetaA = decltype(make_tma_copy(
        GmemTiledCopyMetaA{},
        make_tensor(static_cast<ElementMeta const*>(nullptr), repeat_like(LayoutMetaA{}, int32_t(0)), LayoutMetaA{}),
        SmemLayoutMetaA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}) / Int<NVFP4_META_RDIV>{}, shape<2>(TileShape{}) / Int<NVFP4_META_CDIV>{}),
        _1{}));  // No programmatic multicast
#if NVFP4_META_ON_B
    using TMA_MetaB = decltype(make_tma_copy(
        GmemTiledCopyMetaB{},
        make_tensor(static_cast<ElementMeta const*>(nullptr), repeat_like(LayoutMetaB{}, int32_t(0)), LayoutMetaB{}),
        SmemLayoutMetaB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}) / Int<NVFP4_META_RDIV>{}, shape<2>(TileShape{}) / Int<NVFP4_META_CDIV>{}),
        _1{}));  // No programmatic multicast
#endif

    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_SFA tma_load_sfa;
    TMA_SFB tma_load_sfb;
    TMA_MetaA tma_load_metaa;
#if NVFP4_META_ON_B
    TMA_MetaB tma_load_metab;
#endif
    LayoutSFA layout_SFA;
    LayoutSFB layout_SFB;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    uint32_t tma_transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t tma_transaction_bytes_nk = TmaTransactionBytesNK;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    auto ptr_A = recast_ptr<TmaInternalElementA>(args.ptr_A);
    auto ptr_B = recast_ptr<TmaInternalElementB>(args.ptr_B);

    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,K,L), args.dB));

    Tensor tensor_sfa = make_tensor(args.ptr_SFA, args.layout_SFA);
    Tensor tensor_sfb = make_tensor(args.ptr_SFB, args.layout_SFB);

    Tensor tensor_metaa = make_tensor(args.ptr_MetaA, make_layout(make_shape(M / NVFP4_META_RDIV, K / NVFP4_META_CDIV, L), args.layout_MetaA));
#if NVFP4_META_ON_B
    Tensor tensor_metab = make_tensor(args.ptr_MetaB, make_layout(make_shape(N / NVFP4_META_RDIV, K / NVFP4_META_CDIV, L), args.layout_MetaB));
#endif

    typename Params::TMA_A tma_load_a = make_tma_copy(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{}); // No programmatic multicast
    typename Params::TMA_B tma_load_b = make_tma_copy(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        _1{}); // No programmatic multicast

    typename Params::TMA_SFA tma_load_sfa = make_tma_copy<uint16_t>(
        GmemTiledCopySFA{},
        tensor_sfa,
        SmemLayoutSFA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{}); // No programmatic multicast

    typename Params::TMA_SFB tma_load_sfb = make_tma_copy<uint16_t>(
        GmemTiledCopySFB{},
        tensor_sfb,
        SmemLayoutSFB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShapeSFB{}), shape<2>(TileShapeSFB{})),
        _1{}); // No programmatic multicast

    typename Params::TMA_MetaA tma_load_metaa = make_tma_copy(
        GmemTiledCopyMetaA{},
        tensor_metaa,
        SmemLayoutMetaA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}) / Int<NVFP4_META_RDIV>{}, shape<2>(TileShape{}) / Int<NVFP4_META_CDIV>{}),
        _1{}); // No programmatic multicast
#if NVFP4_META_ON_B
    typename Params::TMA_MetaB tma_load_metab = make_tma_copy(
        GmemTiledCopyMetaB{},
        tensor_metab,
        SmemLayoutMetaB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}) / Int<NVFP4_META_RDIV>{}, shape<2>(TileShape{}) / Int<NVFP4_META_CDIV>{}),
        _1{}); // No programmatic multicast
#endif

    return {
      tma_load_a,
      tma_load_b,
      tma_load_sfa,
      tma_load_sfb,
      tma_load_metaa,
#if NVFP4_META_ON_B
      tma_load_metab,
#endif
      args.layout_SFA,
      args.layout_SFB,
      TmaTransactionBytes,
      TmaTransactionBytesMK,
      TmaTransactionBytesNK
    };
  }

  template<class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    constexpr int tma_alignment_bits_A = cutlass::detail::get_input_alignment_bits<ElementA, IsF8F6F4>();
    constexpr int tma_alignment_bits_B = cutlass::detail::get_input_alignment_bits<ElementB, IsF8F6F4>();

    bool implementable = true;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits_A / cutlass::sizeof_bits<ElementA>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), StrideA{});
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits_B / cutlass::sizeof_bits<ElementB>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_sfa.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_sfb.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_metaa.get_tma_descriptor());
#if NVFP4_META_ON_B
    cute::prefetch_tma_descriptor(params.tma_load_metab.get_tma_descriptor());
#endif
  }

  // Temporary adhoc partitioning for scaling factors.
  template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE constexpr
  auto
  thrfrg_SFA(SFATensor&& sfatensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
  {
    CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});

    using AtomShape_MNK  = typename Atom::Shape_MNK;
    using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;

    auto permutation_mnk = TiledPerm{};
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<0>(permutation_mnk),
                            get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfatensor, t_tile);                 // (PermM,PermK)

    // Tile the tensor for the Atom
    auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomM,AtomK),(RestM,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{},_);           // ((ThrV,FrgV),(RestM,RestK))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<1>(thr_layout_vmnk)),
                                        make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))

    return thr_tensor;
  }

  template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE constexpr
  auto
  thrfrg_SFB(SFBTensor&& sfbtensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
  {
    CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});

    using AtomShape_MNK  = typename Atom::Shape_MNK;
    using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;

    auto permutation_mnk = TiledPerm{};
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<1>(permutation_mnk),
                            get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfbtensor, t_tile);                 // (PermN,PermK)

    // Tile the tensor for the Atom
    auto a_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomN,AtomK),(RestN,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{},_);           // ((ThrV,FrgV),(RestN,RestK))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<2>(thr_layout_vmnk)),
                                        make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK)))
    return thr_tensor;
  }

  template <class SFATensor, class ThrMma>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma)
  {
    using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
    auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(), thrfrg_SFA(sfatensor.layout(),thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFA =  thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
    return make_fragment_like<ValTypeSF>(partition_SFA);
  }

  template <class SFBTensor, class ThrMma>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma)
  {
    using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
    auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(), thrfrg_SFB(sfbtensor.layout(),thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFB =  thr_tensor(thr_vnk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
    return make_fragment_like<ValTypeSF>(partition_SFB);
  }

  template<class TiledMma>
  CUTE_HOST_DEVICE constexpr
  auto
  get_layoutSFA_TV(TiledMma& mma)
  {
    // (M,K) -> (M,K)
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
    auto atile = make_tile(_,
                          make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                                make_stride(               Int<1>{} ,                Int<0>{} )),
                                    _));

    // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    // (thr_idx,val) -> (M,K)
    return thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
  }

  template<class TiledMma>
  CUTE_HOST_DEVICE constexpr
  auto
  get_layoutSFB_TV(TiledMma& mma)
  {
    // (N,K) -> (N,K)
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
    auto btile = make_tile(_,
                          make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                                make_stride(               Int<0>{} ,                Int<1>{} )),
                                    _));

    // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    // (thr_idx,val) -> (M,K)
    return thrfrg_SFB(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
  }

  /// Set up the data needed by this collective for load and mma.
  /// Returns a tuple of tensors. The collective and the kernel layer have the contract
  /// Returned tuple must contain at least two elements, with the first two elements being:
  /// gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  /// gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  /// The rest of the tensors can be specified as needed by this collective.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M, N, K, L] = problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = params.tma_load_a.get_tma_tensor(make_shape(M,K,L));                          // (m,k,l)
    Tensor mB_nkl = params.tma_load_b.get_tma_tensor(make_shape(N,K,L));                          // (n,k,l)
    Tensor mSFA_mkl = params.tma_load_sfa.get_tma_tensor(shape(params.layout_SFA));
    Tensor mSFB_nkl = params.tma_load_sfb.get_tma_tensor(shape(params.layout_SFB));

    // metaA/metaB: native (M or N, K/GROUP_SIZE, L) coordinate space (no SF-style config
    // abstraction), so tile with a matching (M,N,K/GROUP_SIZE) shape rather than
    // TileShape{} itself (whose K component is the *element* K-tile, GROUP_SIZE-x too
    // wide here).
    Tensor mMetaA_mkl = params.tma_load_metaa.get_tma_tensor(make_shape(M / NVFP4_META_RDIV, K / NVFP4_META_CDIV, L));
#if NVFP4_META_ON_B
    Tensor mMetaB_nkl = params.tma_load_metab.get_tma_tensor(make_shape(N / NVFP4_META_RDIV, K / NVFP4_META_CDIV, L));
#endif
    auto TileShapeMeta = make_shape(shape<0>(TileShape{}) / Int<NVFP4_META_RDIV>{}, shape<1>(TileShape{}) / Int<NVFP4_META_RDIV>{},
                                    shape<2>(TileShape{}) / Int<NVFP4_META_CDIV>{});

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});        // (BLK_N,BLK_K,n,k,l)

    Tensor gSFA_mkl = local_tile(mSFA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});    // (TILE_M,TILE_K,m,k,l)
    Tensor gSFB_nkl = local_tile(mSFB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});    // (TILE_N,TILE_K,n,k,l)

    Tensor gMetaA_mkl = local_tile(mMetaA_mkl, TileShapeMeta, make_coord(_,_,_), Step<_1, X,_1>{});  // (BLK_M,BLK_K/GS,m,k,l)
#if NVFP4_META_ON_B
    Tensor gMetaB_nkl = local_tile(mMetaB_nkl, TileShapeMeta, make_coord(_,_,_), Step< X,_1,_1>{});  // (BLK_N,BLK_K/GS,n,k,l)
    return cute::make_tuple(gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl, gMetaA_mkl, gMetaB_nkl);
#else
    return cute::make_tuple(gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl, gMetaA_mkl);
#endif
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class TensorA, class TensorB,
    class TensorSFA, class TensorSFB,
    class TensorMetaA,
#if NVFP4_META_ON_B
    class TensorMetaB,
#endif
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
#if NVFP4_META_ON_B
      cute::tuple<TensorA, TensorB, TensorSFA, TensorSFB, TensorMetaA, TensorMetaB> const& load_inputs,
#else
      cute::tuple<TensorA, TensorB, TensorSFA, TensorSFB, TensorMetaA> const& load_inputs,
#endif
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {

      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)
      Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});  // (BLK_M,BLK_K,PIPE)
      Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});  // (BLK_N,BLK_K,PIPE)
      Tensor sMetaA = make_tensor(make_smem_ptr(shared_tensors.smem_MetaA.begin()), SmemLayoutMetaA{});  // (BLK_M,BLK_K/GS,PIPE)
#if NVFP4_META_ON_B
      Tensor sMetaB = make_tensor(make_smem_ptr(shared_tensors.smem_MetaB.begin()), SmemLayoutMetaB{});  // (BLK_N,BLK_K/GS,PIPE)
#endif

      //
      // Prepare the TMA loads for A, B, SFA, SFB, metaA(/metaB)
      //

#if NVFP4_META_ON_B
      auto [gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl, gMetaA_mkl, gMetaB_nkl] = load_inputs;
#else
      auto [gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl, gMetaA_mkl] = load_inputs;
#endif

      auto block_tma_a = params.tma_load_a.get_slice(0);
      auto block_tma_b = params.tma_load_b.get_slice(0);

      auto block_tma_sfa = params.tma_load_sfa.get_slice(0);
      auto block_tma_sfb = params.tma_load_sfb.get_slice(0);

      auto block_tma_metaa = params.tma_load_metaa.get_slice(0);
#if NVFP4_META_ON_B
      auto block_tma_metab = params.tma_load_metab.get_slice(0);
#endif

      // Partition the inputs based on the current block coordinates.
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;

      auto broadcast_n = make_layout(
        make_shape(Int<size<1>(TileShapeSFB{}) / size<1>(TileShape{})>{},
                   Int<cute::numeric_limits<int>::max()>{}),
        make_stride(_0{}, size<1>(TileShapeSFB{}) / size<1>(TileShape{})));
      Tensor gA =   gA_mkl(_,_,m_coord,_,l_coord);                                                    // (BLK_M,BLK_K,k)
      Tensor gB =   gB_nkl(_,_,n_coord,_,l_coord);                                                    // (BLK_N,BLK_K,k)
      Tensor gSFA = gSFA_mkl(_,_,m_coord,_,l_coord);                                                  // (BLK_M,BLK_K,k)
      Tensor gSFB = gSFB_nkl(_,_,broadcast_n(n_coord),_,l_coord);                                     // (BLK_N,BLK_K,k)
      // metaA/metaB have no SF-style broadcast (plain per-row layout) -- index directly,
      // same as A/B themselves.
      Tensor gMetaA = gMetaA_mkl(_,_,m_coord,_,l_coord);                                              // (BLK_M,BLK_K/GS,k)
#if NVFP4_META_ON_B
      Tensor gMetaB = gMetaB_nkl(_,_,n_coord,_,l_coord);                                              // (BLK_N,BLK_K/GS,k)
#endif

      // Partition source and destination tensors for tma copies
      Tensor tAgA = block_tma_a.partition_S(gA);                                              // (TMA,TMA_M,TMA_K,k)
      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB);                                              // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

      Tensor tAgSFA = block_tma_sfa.partition_S(gSFA);                                        // (TMA,TMA_M,TMA_K,k)
      Tensor tAsSFA = block_tma_sfa.partition_D(sSFA);                                        // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgSFB = block_tma_sfb.partition_S(gSFB);                                        // (TMA,TMA_N,TMA_K,k)
      Tensor tBsSFB = block_tma_sfb.partition_D(sSFB);                                        // (TMA,TMA_N,TMA_K,PIPE)

      Tensor tAgMetaA = block_tma_metaa.partition_S(gMetaA);                                  // (TMA,TMA_M,TMA_K/GS,k)
      Tensor tAsMetaA = block_tma_metaa.partition_D(sMetaA);                                  // (TMA,TMA_M,TMA_K/GS,PIPE)
#if NVFP4_META_ON_B
      Tensor tBgMetaB = block_tma_metab.partition_S(gMetaB);                                  // (TMA,TMA_N,TMA_K/GS,k)
      Tensor tBsMetaB = block_tma_metab.partition_D(sMetaB);                                  // (TMA,TMA_N,TMA_K/GS,PIPE)
#endif

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count) {
        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem for *k_tile_iter
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        copy(params.tma_load_a.with(*tma_barrier), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(params.tma_load_b.with(*tma_barrier), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));

        copy(params.tma_load_sfa.with(*tma_barrier), tAgSFA(_,_,_,*k_tile_iter), tAsSFA(_,_,_,write_stage));
        copy(params.tma_load_sfb.with(*tma_barrier), tBgSFB(_,_,_,*k_tile_iter), tBsSFB(_,_,_,write_stage));

        // metaA(/metaB): extra TMA issue(s), sharing the same mbarrier as A/B/SFA/SFB --
        // this is the actual thing being measured (marginal cost of these extra bulk
        // copies in a real, deeply-pipelined GEMM at full problem size).
        copy(params.tma_load_metaa.with(*tma_barrier), tAgMetaA(_,_,_,*k_tile_iter), tAsMetaA(_,_,_,write_stage));
#if NVFP4_META_ON_B
        copy(params.tma_load_metab.with(*tma_barrier), tBgMetaB(_,_,_,*k_tile_iter), tBsMetaB(_,_,_,write_stage));
#endif

        // Advance k tile
        ++k_tile_iter;
        ++smem_pipe_write;
      }
    }
    __syncwarp();
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all
       * Consumer UNLOCKs), or if the stage was never used
       * then would just be acquired since the phase was
       * still inverted from make_producer_start_state
       */
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgTensorC,
    class BlockCoord
  >
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      [[maybe_unused]] Params const& params,
      BlockCoord const& blk_coord) {
    using namespace cute;

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    clear(accum);

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});         // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});         // (BLK_N,BLK_K,PIPE)
    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});  // (BLK_M,BLK_K,PIPE)
    Tensor sSFB = [&]() {
      if constexpr (size<1>(TileShape{}) >= 128) {
        return make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});  // (BLK_N,BLK_K,PIPE)
      }
      else {
        Tensor temp = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});  // (BLK_SFB_N,BLK_K,PIPE)
        auto n = get<1>(blk_coord);
        return temp(make_coord(_,n % (size<1>(TileShapeSFB{}) / size<1>(TileShape{}))), _, _);
      }
    }();

    //
    // Define C accumulators and A/B partitioning
    //

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    // Allocate fragments and descriptors
    Tensor tCrA = thread_mma.partition_fragment_A(sA(_,_,Int<0>{}));                         // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thread_mma.partition_fragment_B(sB(_,_,Int<0>{}));                         // (MMA,MMA_N,MMA_K)

    Tensor tCrSFA = partition_fragment_SFA(sSFA(_,_,Int<0>{}), thread_mma);                  // (MMA,MMA_M,MMA_K)
    Tensor tCrSFB = partition_fragment_SFB(sSFB(_,_,Int<0>{}), thread_mma);                  // (MMA,MMA_N,MMA_K)

    //
    // Copy from smem to registers
    //

    // A
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(
      as_position_independent_swizzle_tensor(sA));                                      // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                            //      (CPY,CPY_M,CPY_K)

    // B
    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(
      as_position_independent_swizzle_tensor(sB));                                      // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                            //      (CPY,CPY_M,CPY_K)

    // SFA
    auto tile_shape_mnk = tile_shape(tiled_mma);
    auto smem_tiled_copy_SFA = make_tiled_copy_impl(SmemCopyAtomSFA{},
                                                    get_layoutSFA_TV(tiled_mma),
                                                    make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk))
                                                  );
    auto smem_thr_copy_SFA   = smem_tiled_copy_SFA.get_thread_slice(thread_idx);
    Tensor tCsSFA            = smem_thr_copy_SFA.partition_S(
        as_position_independent_swizzle_tensor(sSFA));                                      // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrSFA_copy_view  = smem_thr_copy_SFA.retile_D(tCrSFA);                          //      (CPY,CPY_M,CPY_K)

    // SFB
    auto smem_tiled_copy_SFB = make_tiled_copy_impl(SmemCopyAtomSFB{},
                                                    get_layoutSFB_TV(tiled_mma),
                                                    make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk))
                                                  );
    auto smem_thr_copy_SFB   = smem_tiled_copy_SFB.get_thread_slice(thread_idx);
    Tensor tCsSFB            = smem_thr_copy_SFB.partition_S(
      as_position_independent_swizzle_tensor(sSFB));                                       // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrSFB_copy_view  = smem_thr_copy_SFB.retile_D(tCrSFB);                         //      (CPY,CPY_N,CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));                        // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));                        // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                  // CPY_K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                  // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                    // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                    // PIPE

    CUTE_STATIC_ASSERT_V(size<1>(tCsSFA) == size<1>(tCrSFA_copy_view));                    // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsSFA) == size<2>(tCrSFA_copy_view));                    // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrSFA) == size<1>(accum));                               // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrSFB) == size<2>(accum));                               // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsSFA) == size<2>(tCsSFB));                              // CPY_K
    CUTE_STATIC_ASSERT_V(size<3>(tCsSFA) == size<3>(tCsSFB));                              // PIPE
    CUTE_STATIC_ASSERT_V(size<2>(sA) == size<2>(sSFA));                                    // PIPE
    CUTE_STATIC_ASSERT_V(size<2>(sB) == size<2>(sSFA));                                    // PIPE

    //
    // PIPELINED MAIN LOOP
    //

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);

#if NVFP4_META_SW_CORRECT == 1 || NVFP4_META_SW_CORRECT == 3
    //
    // Software (CUDA-core) application of the 2 extra mantissa bits.
    //
    // Data layout written by the activation quantizer:
    //   * the selected element's FP4 code = top 4 bits of its E2M3 code (sign, exp, m2), so the
    //     tensor-core MMA already accumulates that element's truncated value;
    //   * 1 meta byte per GS-group = idx | lo2 << IB | sign << (IB+2) | exp << (IB+3), where lo2
    //     are the 2 extra mantissa bits and sign/exp duplicate the selected element's FP4 code
    //     (spare bits of the byte, so decoding needs no extra A-operand load).
    // Residual of the selected element: delta = (+-) lo2 * 2^(max(exp,1)-1) / 8   (E2M1 units).
    //   Linear: acc(m,n) += sum_{k in S_A(m)} dA sfa * w4 sfb
    //   QK^T  : acc(m,n) += sum_{k in S_Q(m)} dQ sfa * (k4 + [k in S_K(n)] dK) sfb
    //                     + sum_{k in S_K(n)} q4 sfa * dK sfb
    // Applied per k_block (64 K) right before that k_block's MMA, while its smem stage is live.
    //
    constexpr int GS = NVFP4_META_GROUP_SIZE;
    constexpr int IB = (GS == 8) ? 3 : 2;
    constexpr int MMA_M = decltype(size<1>(accum))::value;
    constexpr int MMA_N = decltype(size<2>(accum))::value;
    constexpr int CR = 2 * MMA_M;                                     // distinct rows per thread
    constexpr int CC = 2 * MMA_N;                                     // distinct cols per thread
    constexpr int KB = size<2>(TileShape{}) / decltype(K_BLOCK_MAX)::value;   // K per k_block
    static_assert(decltype(size<0>(accum))::value == 4, "expects the 16x8 accumulator atom");
    static_assert(KB % SFVecSize == 0 && SFVecSize % GS == 0, "group/scale-block mismatch");

    Tensor sMetaA = make_tensor(make_smem_ptr(shared_tensors.smem_MetaA.begin()), SmemLayoutMetaA{});  // (BLK_M,BLK_K/GS,PIPE)
#if NVFP4_META_ON_B
    Tensor sMetaB = make_tensor(make_smem_ptr(shared_tensors.smem_MetaB.begin()), SmemLayoutMetaB{});  // (BLK_N,BLK_K/GS,PIPE)
#endif
    Tensor tCcC = thread_mma.partition_C(make_identity_tensor(take<0,2>(TileShape{})));            // (MMA,MMA_M,MMA_N) -> (m,n)
    // accum(vc + 2*vr, mm, nn) <-> row r = vr + 2*mm, col c = vc + 2*nn

    // GS-group metadata bytes of one 16-block (16/GS bytes, contiguous and aligned)
    auto load_meta16 = [](auto const& sMeta, int row, int kb, int stage) -> uint32_t {
      uint8_t const* p = &sMeta(row, kb / GS, stage);
      if constexpr (GS == 4) return *reinterpret_cast<uint32_t const*>(p);
      else                   return *reinterpret_cast<uint16_t const*>(p);
    };

#if NVFP4_META_SW_CORRECT == 1 || NVFP4_META_SW_CORRECT == 3
    int row_of[CR], col_of[CC];
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < CR; ++r) row_of[r] = get<0>(tCcC(2 * (r & 1), r >> 1, 0));
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < CC; ++c) col_of[c] = get<1>(tCcC(c & 1, 0, c >> 1));
#endif

#if NVFP4_META_SW_CORRECT == 1
    auto delta_of = [](uint8_t meta) {
      float mag = float((meta >> IB) & 3) * __int_as_float((123 + max((meta >> (IB + 3)) & 3, 1)) << 23);  // lo2 * 2^(max(e,1)-4)
      return ((meta >> (IB + 2)) & 1) ? -mag : mag;
    };
    // Raw FP4 access to a K-major 128x128 e2m1 smem stage: 64 B per row, TMA 64B swizzle
    // (Swizzle<2,4,3> on byte addresses: 16B-chunk bits [4,5] ^= bits [7,8]); even k = low nibble.
    static_assert(size<2>(TileShape{}) == 128 && cute::sizeof_bits_v<SmemAllocTypeA> == 4 &&
                  cute::sizeof_bits_v<SmemAllocTypeB> == 4, "raw smem addressing assumes 128-K e2m1 tiles (SW64)");
    static_assert(SFVecSize == 16, "one 8-byte register pair per 16-element scale block");
    uint8_t const* smemA_raw = reinterpret_cast<uint8_t const*>(&shared_tensors.smem_A);
    uint8_t const* smemB_raw = reinterpret_cast<uint8_t const*>(&shared_tensors.smem_B);
    constexpr int ROWS_A = size<0>(TileShape{});
    constexpr int ROWS_B = size<1>(TileShape{});
    // the 16 FP4 codes of (row, 16-block starting at k) as one 8-byte LDS (never crosses a 16B chunk)
    auto load16 = [](uint8_t const* base, int rows, int row, int k, int stage) {
      int off = stage * rows * 64 + row * 64 + (k >> 1);
      off ^= ((off >> 7) & 3) << 4;
      return *reinterpret_cast<uint2 const*>(base + off);
    };
    // FP4 code at position pos (0..15) of a load16() result -> float (E2M1 units)
    auto e2m1_of = [](uint2 v, int pos) {
      uint32_t w = (pos & 8) ? v.y : v.x;
      uint8_t nib = (w >> ((pos & 7) * 4)) & 0xF;
      return __half2float(__half(__nv_cvt_fp4_to_halfraw(static_cast<__nv_fp4_storage_t>(nib), __NV_E2M1)));
    };

    // Columns are processed CH at a time so that accum + cached operands stay in registers.
    constexpr int CH = (CC >= 8) ? 8 : CC;

    auto correct_kblock = [&](int k_block, int stage) {
      CUTLASS_PRAGMA_NO_UNROLL                                        // bounds register live ranges
      for (int sb = 0; sb < KB / SFVecSize; ++sb) {
        const int kb = k_block * KB + sb * SFVecSize;                 // first K of this scale block
        CUTLASS_PRAGMA_UNROLL
        for (int c0 = 0; c0 < CC; c0 += CH) {
          uint2 wB[CH];                                               // B operand codes, 16 per col
          float sfb[CH];
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < CH; ++c) {
            wB[c]  = load16(smemB_raw, ROWS_B, col_of[c0 + c], kb, stage);
            sfb[c] = float(sSFB(col_of[c0 + c], kb, stage));
          }
#if NVFP4_META_ON_B
          uint32_t mB[CH];                                            // K's metadata bytes (2 per col)
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < CH; ++c) mB[c] = load_meta16(sMetaB, col_of[c0 + c], kb, stage);
#endif
          CUTLASS_PRAGMA_UNROLL
          for (int r = 0; r < CR; ++r) {
            const float sfa = float(sSFA(row_of[r], kb, stage));
            const uint32_t mA = load_meta16(sMetaA, row_of[r], kb, stage);
            CUTLASS_PRAGMA_UNROLL
            for (int g = 0; g < SFVecSize / GS; ++g) {
              const uint8_t ma  = (mA >> (8 * g)) & 0xFF;
              const int     pos = g * GS + (ma & (GS - 1));
              const float   da  = delta_of(ma) * sfa;
              CUTLASS_PRAGMA_UNROLL
              for (int c = 0; c < CH; ++c) {
                float w = e2m1_of(wB[c], pos);
#if NVFP4_META_ON_B
                const uint8_t mb = (mB[c] >> (8 * g)) & 0xFF;
                if ((mb & (GS - 1)) == (ma & (GS - 1))) w += delta_of(mb);   // K promoted at the same k
#endif
                const int cc = c0 + c;
                float& acc = accum(2 * (r & 1) + (cc & 1), r >> 1, cc >> 1);
                acc = fmaf(da, w * sfb[c], acc);
              }
            }
#if NVFP4_META_ON_B
            const uint2 qA = load16(smemA_raw, ROWS_A, row_of[r], kb, stage);
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < CH; ++c) {
              CUTLASS_PRAGMA_UNROLL
              for (int g = 0; g < SFVecSize / GS; ++g) {
                const uint8_t mb = (mB[c] >> (8 * g)) & 0xFF;
                float q = e2m1_of(qA, g * GS + (mb & (GS - 1)));
                const int cc = c0 + c;
                float& acc = accum(2 * (r & 1) + (cc & 1), r >> 1, cc >> 1);
                acc = fmaf(q * sfa, delta_of(mb) * sfb[c], acc);
              }
            }
#endif
          }
        }
      }
    };
#endif  // NVFP4_META_SW_CORRECT == 1

#if NVFP4_META_SW_CORRECT == 3
    // Per-thread register accumulator -- pack_kblock() XORs into this every k_block
    // instead of touching global memory on every call; mma() flushes ONE store per
    // CTA-tile at the very end (see below), isolating the packing compute itself
    // from the cost of a per-k_block global store.
    uint32_t pack_accum = 0;
#ifndef NVFP4_META_TIE_OPERAND
#define NVFP4_META_TIE_OPERAND 0
#endif
    // NVFP4_META_TIE_OPERAND=1 (simulation build only): without a consumer the
    // packing is scheduled after the k_block's MMAs. To get the schedule of an
    // instruction that reads the packed register, each MMA's accumulator (its C/D
    // register group) gets an in-place, value-preserving
    //   lop3 acc, acc, packed, tie_zero, 0x78    // acc ^ (packed & 0)
    // per packed operand (A's, then B's for QK^T). tie_zero is 0 at run time but
    // opaque to nvcc/ptxas, so ptxas keeps a real data dependency packed -> acc ->
    // OMMA. The accumulator is updated in place by the OMMAs anyway, so removing
    // these LOP3s from the trace leaves the OMMA->OMMA accumulator chain intact; the
    // trace post-processor drops them and appends `packed` to the OMMA whose C
    // register they wrote (13th operand, 14th for B).
    uint32_t packedA_kb[MMA_M];
    uint32_t packedB_kb[MMA_N];
    const uint32_t tie_zero = uint32_t(k_tile_count) >> 31;   // k_tile_count >= 0
    // A's metadata operand: one packed value PER M ATOM (MMA_M of them per k_block),
    // not just one for the whole warp -- real SASS uses a different SFA register per
    // M atom for exactly the same reason (A data differs by M atom), so the 13th
    // operand's content must too; each of the warp's OMMAs reads whichever matches
    // its own M atom. For GS==4 (Linear) one row's load_meta16() already yields the
    // full 4 values (matches a[0..3]); for GS==8 (QK^T) it yields 2, so 2 distinct
    // rows of the SAME M atom (row_of[2*mm], row_of[2*mm+1]) combine to the 4 needed.
    // B's metadata (QK^T, NVFP4_META_ON_B only) is a SEPARATE 14th operand, one
    // packed value per N ATOM (MMA_N of them) -- see the big comment above this mma()
    // for why it's kept separate rather than merged into A's 13th operand.
    // GS==16 (QK^T, 1-of-16): each thread's A fragment a[0..3] covers rows r, r+8 and
    // K ranges kb+8*ai and kb+32+8*ai (ai = lane%4) of the 16x64 atom, i.e. exactly one
    // 16-group per (row, half); B's b[0..1] the same two K ranges of column lane/4. The
    // packed value holds those groups' 6-bit metadata (4-bit index + 2-bit correction):
    // A 4 x 6 = 24 bits, B 2 x 6 = 12 bits. Bytes are read through the smem tensor so
    // the transposed (M-contiguous) metadata layout works too.
    const int lane_ai = threadIdx.x & 3;
    const int lane_bj = (threadIdx.x & 31) >> 2;
#if NVFP4_META_PAIR16
    // 32-bit word holding groups (r, glo), (r+8, glo), (r, ghi), (r+8, ghi) of the k_block
    // starting at kb, for this lane (glo = (kb + 8*ai)/16, ghi = glo + 2); r % 16 < 8.
    auto pair16_word = [&](uint8_t const* stage_base, int r, int kb) -> uint32_t {
      constexpr int ROW_BYTES = decltype(size<2>(TileShape{}))::value / 8;
      const int prow = (r / 16) * 8 + (r % 8);
      const int byte = (kb / 64) * 8 + (lane_ai >> 1) * 4;
      return *reinterpret_cast<uint32_t const*>(stage_base + prow * ROW_BYTES + byte);
    };
#endif
    auto pack_kblock = [&](int k_block, int stage) {
      const int kb = k_block * KB;              // first K of this k_block (one sb/SFVecSize-block)
      CUTLASS_PRAGMA_UNROLL
      for (int mm = 0; mm < MMA_M; ++mm) {
        uint32_t packed = 0;
        if constexpr (GS == 16) {
#if NVFP4_META_PAIR16
          // one 32-bit load: bytes [r lo, r+8 lo, r hi, r+8 hi] = a[0..3]'s groups
          const int r0 = row_of[2 * mm];
          const uint32_t w = pair16_word(&sMetaA(0, 0, stage), r0, kb);
          packed = (w & 0x3F) | ((w >> 8) & 0x3F) << 6 | ((w >> 16) & 0x3F) << 12 | ((w >> 24) & 0x3F) << 18;
#else
          const int glo = (kb + 8 * lane_ai) / 16, ghi = (kb + 32 + 8 * lane_ai) / 16;
          const int r0 = row_of[2 * mm], r1 = row_of[2 * mm + 1];
          packed = (uint32_t)(sMetaA(r0, glo, stage) & 0x3F)
                 | (uint32_t)(sMetaA(r1, glo, stage) & 0x3F) << 6
                 | (uint32_t)(sMetaA(r0, ghi, stage) & 0x3F) << 12
                 | (uint32_t)(sMetaA(r1, ghi, stage) & 0x3F) << 18;
#endif
        } else if constexpr (GS == 4) {
          const uint32_t mA = load_meta16(sMetaA, row_of[2 * mm], kb, stage);   // 4 x 4-bit values
          CUTLASS_PRAGMA_UNROLL
          for (int g = 0; g < 4; ++g) packed |= (uint32_t)((mA >> (8 * g)) & 0x0F) << (4 * g);
        } else if constexpr (NVFP4_META_PREPACK) {
          // one 32-bit load: the stored word is the operand
          static_assert(KB == 64, "PREPACK assumes 64-K k_blocks");
          const int r0 = row_of[2 * mm];
          packed = *reinterpret_cast<uint32_t const*>(&sMetaA((r0 >> 4) * 8 + (r0 & 7), k_block * 16 + lane_ai * 4, stage));
        } else {
          static_assert(GS == 8, "GS must be 4, 8 or 16");
          const uint32_t mA0 = load_meta16(sMetaA, row_of[2 * mm],     kb, stage);   // 2 x 5-bit values
          const uint32_t mA1 = load_meta16(sMetaA, row_of[2 * mm + 1], kb, stage);   // 2 x 5-bit values
          CUTLASS_PRAGMA_UNROLL
          for (int g = 0; g < 2; ++g) {
            packed |= (uint32_t)((mA0 >> (8 * g)) & 0x1F) << (5 * g);
            packed |= (uint32_t)((mA1 >> (8 * g)) & 0x1F) << (5 * (2 + g));
          }
        }
        // Not fed to any MMA (no real instruction has a 13th operand) -- XOR into the
        // per-thread register accumulator (flushed once at the end of mma()) purely so
        // the compiler can't prove this computation is unused and elide it.
        pack_accum ^= packed;
        packedA_kb[mm] = packed;
      }
#if NVFP4_META_ON_B
      CUTLASS_PRAGMA_UNROLL
      for (int nn = 0; nn < MMA_N; ++nn) {
        uint32_t packedB = 0;
        if constexpr (GS == 16) {
          const int c = col_of[2 * nn] - 2 * lane_ai + lane_bj;   // this thread's B column
#if NVFP4_META_PAIR16
          // one 32-bit load of the row pair holding column c; its lo/hi bytes for c
          const uint32_t w = pair16_word(&sMetaB(0, 0, stage), c & ~8, kb) >> (8 * ((c >> 3) & 1));
          packedB = (w & 0x3F) | ((w >> 16) & 0x3F) << 6;
#else
          const int glo = (kb + 8 * lane_ai) / 16, ghi = (kb + 32 + 8 * lane_ai) / 16;
          packedB = (uint32_t)(sMetaB(c, glo, stage) & 0x3F) | (uint32_t)(sMetaB(c, ghi, stage) & 0x3F) << 6;
#endif
        } else {
          const uint32_t mB = load_meta16(sMetaB, col_of[2 * nn], kb, stage);   // 2 x 5-bit values
          CUTLASS_PRAGMA_UNROLL
          for (int g = 0; g < 2; ++g) packedB |= (uint32_t)((mB >> (8 * g)) & 0x1F) << (5 * g);
        }
        pack_accum ^= packedB;
        packedB_kb[nn] = packedB;
      }
#endif
    };
#endif  // NVFP4_META_SW_CORRECT == 3
#endif

    int read_stage = smem_pipe_read.index();
    auto tCsA_stage   = tCsA(_,_,_,read_stage);
    auto tCsB_stage   = tCsB(_,_,_,read_stage);
    auto tCsSFA_stage = tCsSFA(_,_,_,read_stage);
    auto tCsSFB_stage = tCsSFB(_,_,_,read_stage);

#if NVFP4_META_SW_CORRECT == 2
    //
    // Software (tensor-core) application of the 2 extra mantissa bits.
    //
    // x_ext = x4 + d with x4 = truncated E2M3 (already in the A/B fragments) and
    // d = (+-) lo2 * 2^(max(e,1)-1) / 8 in E2M1 units. 4*d takes values in {0.5,1,1.5,2,3,4,6},
    // i.e. d is EXACTLY an E2M1 code under scale sf/4. So the residual operand R (zero except at
    // the selected positions) is a valid NVFP4 operand and A_ext*B = A4*B + R_A*B runs on the FP4
    // tensor core. R fragments are built per k_block in registers from the smem metadata (no
    // extra global traffic), reusing the already-loaded B/SFB fragments.
    //
    static_assert(NVFP4_META_GROUP_SIZE == 4 || NVFP4_META_GROUP_SIZE == 8, "group size");
    constexpr int GS = NVFP4_META_GROUP_SIZE;
    constexpr int IB = (GS == 8) ? 3 : 2;
    constexpr int BLK_K_META = size<2>(TileShape{}) / GS;              // meta bytes per row per stage
    // 4*d -> E2M1 code, indexed by lo2*3 + (max(e,1)-1)
    constexpr uint64_t kDeltaCodeLut =
        (0ull << 0)  | (0ull << 3)  | (0ull << 6)  |
        (1ull << 9)  | (2ull << 12) | (4ull << 15) |
        (2ull << 18) | (4ull << 21) | (6ull << 24) |
        (3ull << 27) | (5ull << 30) | (7ull << 33);
#if NVFP4_META_DCODE
    // Quantizer already stored the residual's E2M1 code: meta = idx | dcode << IB.
    auto delta_code = [](uint32_t meta) -> uint32_t { return (meta >> IB) & 0xF; };
#else
    auto delta_code = [](uint32_t meta) -> uint32_t {
      uint32_t lo = (meta >> IB) & 3, e = (meta >> (IB + 3)) & 3;
      uint32_t E  = (e > 0 ? e : 1) - 1;
      uint32_t c  = uint32_t(kDeltaCodeLut >> (3 * (lo * 3 + E))) & 7;
      return c | (((meta >> (IB + 2)) & 1) << 3);
    };
#endif
    // Residual fragment for one k_block: every 32-bit fragment register holds 8 K-consecutive
    // codes of one row (k % 8 == 0), so it is assembled from 8/GS metadata bytes.
    auto build_residual = [&](auto const& coords, auto&& rfrag, uint8_t const* meta, int rows, int stage) {
      auto r32 = recast<uint32_t>(rfrag);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(r32); ++j) {
        auto mk = coords(8 * j);
        uint8_t const* mp = meta + (stage * rows + get<0>(mk)) * BLK_K_META + get<1>(mk) / GS;
        uint32_t r = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int g = 0; g < 8 / GS; ++g) {
          uint32_t mb = mp[g];
          r |= delta_code(mb) << (4 * (g * GS + (mb & (GS - 1))));
        }
        r32(j) = r;
      }
    };
    // sf/4 on the raw UE4M3 bits: exponent field - 2 (exact while the result stays normal);
    // subnormal results fall back to a float round trip.
    auto quarter_sf = [](auto const& sf, auto&& out) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(sf); ++i) {
        uint8_t b = sf(i).raw();
        out(i) = ((b >> 3) & 0xF) >= 3 ? ElementSF::bitcast(uint8_t(b - 0x10)) : ElementSF(float(sf(i)) * 0.25f);
      }
    };
    uint8_t const* smemMetaA_raw = reinterpret_cast<uint8_t const*>(&shared_tensors.smem_MetaA);
    Tensor tCcA   = thread_mma.partition_A(make_identity_tensor(select<0,2>(TileShape{})));   // (MMA,MMA_M,MMA_K) -> (m,k)
    Tensor tCrRA  = make_fragment_like(tCrA);                                                 // residual A
    Tensor tCrRSFA = make_fragment_like(tCrSFA);                                              // sfa/4
#if NVFP4_META_ON_B
    uint8_t const* smemMetaB_raw = reinterpret_cast<uint8_t const*>(&shared_tensors.smem_MetaB);
    Tensor tCcB   = thread_mma.partition_B(make_identity_tensor(select<1,2>(TileShape{})));   // (MMA,MMA_N,MMA_K) -> (n,k)
    Tensor tCrRB  = make_fragment_like(tCrB);
    Tensor tCrRSFB = make_fragment_like(tCrSFB);
#endif
#endif

    auto copy_kblock = [&](auto k_block) {
        // copy smem->rmem for A/B operand
      copy(smem_tiled_copy_A, tCsA_stage(_,_,k_block), tCrA_copy_view(_,_,k_block));
      copy(smem_tiled_copy_B, tCsB_stage(_,_,k_block), tCrB_copy_view(_,_,k_block));

      // Left shift A,B for FP4
      using MMAOp = typename TiledMma::MMA_Op;
      fp4_shift_A(MMAOp{}, tCrA_copy_view(_,_,k_block));
      fp4_shift_B(MMAOp{}, tCrB_copy_view(_,_,k_block));


      // Copy smem->rmem for SFA/SFB operand
      copy(tCsSFA_stage(_,_,k_block), tCrSFA_copy_view(_,_,k_block));
      copy(tCsSFB_stage(_,_,k_block), tCrSFB_copy_view(_,_,k_block));
#if NVFP4_META_SW_CORRECT == 2
      build_residual(tCcA(_,_,k_block), tCrRA(_,_,k_block), smemMetaA_raw, size<0>(TileShape{}), read_stage);
      quarter_sf(tCrSFA(_,_,k_block), tCrRSFA(_,_,k_block));
#if NVFP4_META_ON_B
      build_residual(tCcB(_,_,k_block), tCrRB(_,_,k_block), smemMetaB_raw, size<1>(TileShape{}), read_stage);
      quarter_sf(tCrSFB(_,_,k_block), tCrRSFB(_,_,k_block));
#endif
#endif
    };

    auto gemm_kblock = [&](auto k_block) {
#if NVFP4_META_SW_CORRECT == 3 && NVFP4_META_TIE_OPERAND
      CUTLASS_PRAGMA_UNROLL
      for (int mm = 0; mm < MMA_M; ++mm) {
        CUTLASS_PRAGMA_UNROLL
        for (int nn = 0; nn < MMA_N; ++nn) {
          asm volatile("lop3.b32 %0, %0, %1, %2, 0x78;" : "+f"(accum(0, mm, nn)) : "r"(packedA_kb[mm]), "r"(tie_zero));
#if NVFP4_META_ON_B
          asm volatile("lop3.b32 %0, %0, %1, %2, 0x78;" : "+f"(accum(0, mm, nn)) : "r"(packedB_kb[nn]), "r"(tie_zero));
#endif
        }
      }
#endif
      // (V,M) x (V,N) => (V,M,N)
      cute::gemm(tiled_mma, make_zip_tensor(tCrA(_,_,k_block), tCrSFA(_,_,k_block)), make_zip_tensor(tCrB(_,_,k_block), tCrSFB(_,_,k_block)), accum);
#if NVFP4_META_SW_CORRECT == 2
      cute::gemm(tiled_mma, make_zip_tensor(tCrRA(_,_,k_block), tCrRSFA(_,_,k_block)), make_zip_tensor(tCrB(_,_,k_block), tCrSFB(_,_,k_block)), accum);
#if NVFP4_META_ON_B
      cute::gemm(tiled_mma, make_zip_tensor(tCrA(_,_,k_block), tCrSFA(_,_,k_block)), make_zip_tensor(tCrRB(_,_,k_block), tCrRSFB(_,_,k_block)), accum);
      cute::gemm(tiled_mma, make_zip_tensor(tCrRA(_,_,k_block), tCrRSFA(_,_,k_block)), make_zip_tensor(tCrRB(_,_,k_block), tCrRSFB(_,_,k_block)), accum);
#endif
#endif
    };

    pipeline.consumer_wait(smem_pipe_read);

    copy_kblock(_0{});
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 1; --k_tile_count) {
      //
      // Compute on k_tile
      //
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {

        auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

#if NVFP4_META_SW_CORRECT == 1
        correct_kblock(k_block, read_stage);   // smem stage still live (released just below)
#elif NVFP4_META_SW_CORRECT == 3
        pack_kblock(k_block, read_stage);
#endif
        if (k_block == K_BLOCK_MAX - 1) {
          cutlass::arch::NamedBarrier::sync(
          thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
          // UNLOCK smem_pipe_read, done _computing_ on it
          pipeline.consumer_release(smem_pipe_read);
          ++smem_pipe_read;
          read_stage = smem_pipe_read.index();
          tCsA_stage   = tCsA(_,_,_,read_stage);
          tCsB_stage   = tCsB(_,_,_,read_stage);
          tCsSFA_stage = tCsSFA(_,_,_,read_stage);
          tCsSFB_stage = tCsSFB(_,_,_,read_stage);
          pipeline.consumer_wait(smem_pipe_read);
        }

        copy_kblock(k_block_next);
        gemm_kblock(k_block);

      });
    } // k_tile_count

    //
    // Hoist out last k_tile
    //
    for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {

      auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

#if NVFP4_META_SW_CORRECT == 1
      correct_kblock(k_block, read_stage);
#elif NVFP4_META_SW_CORRECT == 3
      pack_kblock(k_block, read_stage);
#endif
      if (k_block == K_BLOCK_MAX - 1) {
        cutlass::arch::NamedBarrier::sync(
        thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
        // UNLOCK smem_pipe_read, done _computing_ on it
        pipeline.consumer_release(smem_pipe_read);
        ++smem_pipe_read;
      }

      if (k_block_next > 0) {
        copy_kblock(k_block_next);
      }
      gemm_kblock(k_block);

    });

#if NVFP4_META_SW_CORRECT == 3
    if ((threadIdx.x & 31) == 0) g_nvfp4_pack_sink[blockIdx.x % NVFP4_PACK_SINK_SLOTS] = pack_accum;
#endif
}

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline, PipelineState, int) {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
