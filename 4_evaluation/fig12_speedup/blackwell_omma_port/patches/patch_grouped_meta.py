#!/usr/bin/env python3
"""Adds Crescent correction-data (metaA) support to the grouped (ptr-array) SM120 block-scaled
collective, cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp. Everything is guarded by
NVFP4_META_GROUPED (default 0 -> the file compiles exactly like the unpatched one).

Linear configuration only (MoE expert GEMMs): correction data on A (activations), 1 byte per
NVFP4_META_GROUP_SIZE-element K-group, row-major (M_g, K_g/GS) per group, its own pointer per
group (ptr_MetaA[g]); TMA-loaded into the same pipeline stage / mbarrier as A/B/SFA/SFB, with a
fifth per-SM TMA descriptor that is re-pointed and re-shaped on every group switch exactly like
A's. NVFP4_META_SW_CORRECT=0: load only; =3: + the same packing as the non-grouped collective.

Usage: patch_grouped_meta.py <orig.hpp> <out.hpp>
"""
import sys

src = open(sys.argv[1]).read()


def rep(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:80])
    src = src.replace(old, new)


G = "#if NVFP4_META_GROUPED\n"
E = "#endif\n"

# ---- config + sink -------------------------------------------------------------------------
rep('''#include "cute/numeric/arithmetic_tuple.hpp"
''', '''#include "cute/numeric/arithmetic_tuple.hpp"

// Crescent correction data for the grouped (MoE expert) GEMM -- see patches/patch_grouped_meta.py.
#ifndef NVFP4_META_GROUPED
#define NVFP4_META_GROUPED 0
#endif
#if NVFP4_META_GROUPED
#ifndef NVFP4_META_GROUP_SIZE
#define NVFP4_META_GROUP_SIZE 8
#endif
#ifndef NVFP4_META_SW_CORRECT
#define NVFP4_META_SW_CORRECT 0
#endif
#ifndef NVFP4_META_TIE_OPERAND
#define NVFP4_META_TIE_OPERAND 0
#endif
// NVFP4_META_PREPACK=1 (GS=8): the quantizer stores correction data already in the 13th-operand
// format. For OMMA m16n8k64 a thread (ai = lane%4) holds rows r, r+8 (r%16 < 8) and 8-element
// groups ai, 4+ai of each 64-K k_block; their four 5-bit values form one 32-bit word, one value
// per byte (so the quantizer writes them with plain byte stores from different rows' threads)
//   w = c(r,ai) | c(r,4+ai) << 8 | c(r+8,ai) << 16 | c(r+8,4+ai) << 24
// stored at physical row p = (r/16)*8 + r%8, byte (k/64)*16 + ai*4 of a (M/2, K/4)-byte row-major
// tensor (same total size as the plain (M, K/8) layout). The GEMM reads it with one 32-bit LDS.
#ifndef NVFP4_META_PREPACK
#define NVFP4_META_PREPACK 0
#endif
#if NVFP4_META_SW_CORRECT == 3
// one slot per CTA (plain store, no atomics) -- keeps the packing from being dead code
__device__ uint32_t g_nvfp4_pack_sink_grp[2048];
#endif
#endif
''')

# ---- types / smem layout -------------------------------------------------------------------
rep('''  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
''', '''  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");

#if NVFP4_META_GROUPED
  using ElementMeta = uint8_t;
  static constexpr int MetaGS = NVFP4_META_GROUP_SIZE;
  static_assert(!NVFP4_META_PREPACK || MetaGS == 8, "PREPACK: GS=8 only");
  static constexpr int MetaRD = NVFP4_META_PREPACK ? 2 : 1;        // rows   -> rows / MetaRD
  static constexpr int MetaCD = NVFP4_META_PREPACK ? 4 : MetaGS;   // K      -> K / MetaCD bytes
  // per-group correction data: (M_g, K_g/GS, 1), K/GS contiguous
#ifndef NVFP4_META_TRANSPOSE
#define NVFP4_META_TRANSPOSE 0
#endif
  // NVFP4_META_TRANSPOSE=1: (M_g, K_g/GS) stored M-contiguous (needed when K/GS rows are < 16 B,
  // e.g. the NVFP6 extra-scale emulation, GS=16: 8 B per row per 128-K tile)
  static_assert(!(NVFP4_META_TRANSPOSE && NVFP4_META_PREPACK), "TRANSPOSE xor PREPACK");
#if NVFP4_META_TRANSPOSE
  using InternalLayoutMetaA = cute::Stride<cute::Int<1>, int64_t, int64_t>;
  CUTE_HOST_DEVICE static InternalLayoutMetaA meta_stride(int64_t rows, int64_t cols) { return {_1{}, rows, rows * cols}; }
#else
  using InternalLayoutMetaA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  CUTE_HOST_DEVICE static InternalLayoutMetaA meta_stride(int64_t rows, int64_t cols) { return {cols, _1{}, rows * cols}; }
#endif
  using SmemLayoutAtomMetaA = decltype(make_layout(
    make_shape(shape<0>(TileShape{}) / Int<MetaRD>{}, shape<2>(TileShape{}) / Int<MetaCD>{}),
#if NVFP4_META_TRANSPOSE
    make_stride(Int<1>{}, shape<0>(TileShape{}))));
#else
    make_stride(shape<2>(TileShape{}) / Int<MetaCD>{}, Int<1>{})));
#endif
  using SmemLayoutMetaA = decltype(make_layout(
    append(shape(SmemLayoutAtomMetaA{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomMetaA{}), size(SmemLayoutAtomMetaA{}))));
#endif
''')

rep('''    cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutA{})) * sizeof_bits<ElementA>::value));

  static constexpr uint32_t TmaTransactionBytesNK''', '''    cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutA{})) * sizeof_bits<ElementA>::value)
#if NVFP4_META_GROUPED
    + cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutMetaA{})) * cute::sizeof_bits_v<ElementMeta>)
#endif
    );

  static constexpr uint32_t TmaTransactionBytesNK''')

rep('''      alignas(16)   cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
    } tensors;''', '''      alignas(16)   cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
#if NVFP4_META_GROUPED
      alignas(16)   cute::ArrayEngine<ElementMeta, cute::cosize_v<SmemLayoutMetaA>> smem_MetaA;
#endif
    } tensors;''')

rep('''      cute::TmaDescriptor smem_tensormap_SFB;
    } tensormaps;''', '''      cute::TmaDescriptor smem_tensormap_SFB;
#if NVFP4_META_GROUPED
      cute::TmaDescriptor smem_tensormap_MetaA;
#endif
    } tensormaps;''')

# ---- Arguments / Params --------------------------------------------------------------------
rep('''    ElementSF const** ptr_SFB{nullptr};
    LayoutSFB layout_SFB{};
  };''', '''    ElementSF const** ptr_SFB{nullptr};
    LayoutSFB layout_SFB{};
#if NVFP4_META_GROUPED
    ElementMeta const** ptr_MetaA{nullptr};   // per group, (M_g, K_g/GS) row-major
#endif
  };''')

rep('''    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_SFA tma_load_sfa;
    TMA_SFB tma_load_sfb;
''', '''#if NVFP4_META_GROUPED
    using TMA_MetaA = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(static_cast<ElementMeta const*>(nullptr), repeat_like(InternalLayoutMetaA{}, int32_t(0)), InternalLayoutMetaA{}),
        SmemLayoutMetaA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}) / Int<MetaRD>{}, shape<2>(TileShape{}) / Int<MetaCD>{}),
        _1{}));
#endif

    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_SFA tma_load_sfa;
    TMA_SFB tma_load_sfb;
#if NVFP4_META_GROUPED
    TMA_MetaA tma_load_metaa;
#endif
''')

rep('''    ElementSF const** ptr_SFB;
    LayoutSFB layout_SFB;
  };''', '''    ElementSF const** ptr_SFB;
    LayoutSFB layout_SFB;
#if NVFP4_META_GROUPED
    ElementMeta const** ptr_MetaA;
#endif
  };''')

rep('''        make_shape(shape<1>(TileShapeSFB{}), shape<2>(TileShapeSFB{})),
        _1{}); // No programmatic multicast

    return {
      tma_load_a,
      tma_load_b,
      tma_load_sfa,
      tma_load_sfb,
''', '''        make_shape(shape<1>(TileShapeSFB{}), shape<2>(TileShapeSFB{})),
        _1{}); // No programmatic multicast

#if NVFP4_META_GROUPED
    // placeholder (TileM, TileK/GS) shape; replaced per group before the first load
    Tensor tensor_metaa = make_tensor(static_cast<ElementMeta const*>(nullptr),
        make_layout(make_shape(init_M / MetaRD, init_K / MetaCD, init_L),
                    meta_stride(init_M / MetaRD, init_K / MetaCD)));
    typename Params::TMA_MetaA tma_load_metaa = make_tma_copy(
        SM90_TMA_LOAD{},
        tensor_metaa,
        SmemLayoutMetaA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}) / Int<MetaRD>{}, shape<2>(TileShape{}) / Int<MetaCD>{}),
        _1{});
#endif

    return {
      tma_load_a,
      tma_load_b,
      tma_load_sfa,
      tma_load_sfb,
#if NVFP4_META_GROUPED
      tma_load_metaa,
#endif
''')

rep('''      reinterpret_cast<ElementSF const**>(args.ptr_SFB),
      args.layout_SFB
    };''', '''      reinterpret_cast<ElementSF const**>(args.ptr_SFB),
      args.layout_SFB
#if NVFP4_META_GROUPED
      , args.ptr_MetaA
#endif
    };''')

rep('''    constexpr uint32_t NumInputTensors = 4;''', '''    constexpr uint32_t NumInputTensors = 4 + NVFP4_META_GROUPED;''')

# ---- load_init / load ----------------------------------------------------------------------
rep('''    Tensor gSFB_nkl = local_tile(mSFB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});    // (TILE_N,TILE_K,n,k,l)

    return cute::make_tuple(gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl);''', '''    Tensor gSFB_nkl = local_tile(mSFB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});    // (TILE_N,TILE_K,n,k,l)

#if NVFP4_META_GROUPED
    Tensor mMetaA_mkl = params.tma_load_metaa.get_tma_tensor(make_shape(M / MetaRD, K / MetaCD, init_L));
    auto TileShapeMeta = make_shape(shape<0>(TileShape{}) / Int<MetaRD>{}, shape<1>(TileShape{}), shape<2>(TileShape{}) / Int<MetaCD>{});
    Tensor gMetaA_mkl = local_tile(mMetaA_mkl, TileShapeMeta, make_coord(_,_,_), Step<_1, X,_1>{});  // (BLK_M,BLK_K/GS,m,k,l)
    return cute::make_tuple(gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl, gMetaA_mkl);
#else
    return cute::make_tuple(gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl);
#endif''')

# load(): generic tuple types
rep('''  template <
    class TensorA, class TensorB,
    class TensorSFA, class TensorSFB,
    class TensorMapA, class TensorMapB,
    class TensorMapSFA, class TensorMapSFB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB, TensorSFA, TensorSFB> const& load_inputs,
      cute::tuple<TensorMapA, TensorMapB, TensorMapSFA, TensorMapSFB> const& input_tensormaps,''', '''  template <
    class LoadInputs,
    class InputTensorMaps,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      LoadInputs const& load_inputs,
      InputTensorMaps const& input_tensormaps,''')

rep('''      auto [gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl] = load_inputs;
''', '''      auto gA_mkl = get<0>(load_inputs);
      auto gB_nkl = get<1>(load_inputs);
      auto gSFA_mkl = get<2>(load_inputs);
      auto gSFB_nkl = get<3>(load_inputs);
''')

rep('''      Tensor tBsSFB = block_tma_sfb.partition_D(sSFB);                                        // (TMA,TMA_N,TMA_K,PIPE)

      // Mainloop''', '''      Tensor tBsSFB = block_tma_sfb.partition_D(sSFB);                                        // (TMA,TMA_N,TMA_K,PIPE)

#if NVFP4_META_GROUPED
      Tensor sMetaA = make_tensor(make_smem_ptr(shared_tensors.smem_MetaA.begin()), SmemLayoutMetaA{});  // (BLK_M,BLK_K/GS,PIPE)
      auto block_tma_metaa = params.tma_load_metaa.get_slice(0);
      Tensor gMetaA = get<4>(load_inputs)(_,_,m_coord,_,l_coord);                                       // (BLK_M,BLK_K/GS,k)
      Tensor tAgMetaA = block_tma_metaa.partition_S(gMetaA);
      Tensor tAsMetaA = block_tma_metaa.partition_D(sMetaA);
#endif

      // Mainloop''')

rep('''        copy(params.tma_load_sfb.with(get<3>(input_tensormaps),*tma_barrier), tBgSFB(_,_,_,*k_tile_iter), tBsSFB(_,_,_,write_stage));
''', '''        copy(params.tma_load_sfb.with(get<3>(input_tensormaps),*tma_barrier), tBgSFB(_,_,_,*k_tile_iter), tBsSFB(_,_,_,write_stage));
#if NVFP4_META_GROUPED
        // correction data: same stage, same mbarrier as A/B/SFA/SFB
        copy(params.tma_load_metaa.with(get<4>(input_tensormaps),*tma_barrier), tAgMetaA(_,_,_,*k_tile_iter), tAsMetaA(_,_,_,write_stage));
#endif
''')

# ---- mma(): packing (mode 3), same as the non-grouped collective's GS==8 Linear path --------
rep('''    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);

    int read_stage = smem_pipe_read.index();''', '''    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);

#if NVFP4_META_GROUPED && NVFP4_META_SW_CORRECT == 3
    constexpr int GS = MetaGS;
    static_assert(GS == 8, "grouped packing: Linear 1-of-8 only");
    constexpr int MMA_M = decltype(size<1>(accum))::value;
    constexpr int CR = 2 * MMA_M;
    constexpr int KB = size<2>(TileShape{}) / decltype(K_BLOCK_MAX)::value;   // K per k_block
    Tensor sMetaA = make_tensor(make_smem_ptr(shared_tensors.smem_MetaA.begin()), SmemLayoutMetaA{});  // (BLK_M,BLK_K/GS,PIPE)
    Tensor tCcC = thread_mma.partition_C(make_identity_tensor(take<0,2>(TileShape{})));            // (MMA,MMA_M,MMA_N) -> (m,n)
    int row_of[CR];
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < CR; ++r) row_of[r] = get<0>(tCcC(2 * (r & 1), r >> 1, 0));
    // the 16/GS correction bytes of one 16-element block of row `row`
    auto load_meta16 = [&](int row, int kb, int stage) -> uint32_t {
      return *reinterpret_cast<uint16_t const*>(&sMetaA(row, kb / GS, stage));
    };
    uint32_t pack_accum = 0;
    uint32_t packedA_kb[MMA_M];
    const uint32_t tie_zero = uint32_t(k_tile_count) >> 31;   // 0 at run time, opaque to the compiler
    // one packed 13th-operand value per M atom per k_block: 2 rows x 2 groups x 5 bits
    auto pack_kblock = [&](int k_block, int stage) {
#if NVFP4_META_PREPACK
      // one 32-bit load per M atom: the stored word IS the operand
      static_assert(KB == 64, "PREPACK assumes 64-K k_blocks");
      const int lane_ai = threadIdx.x & 3;
      CUTLASS_PRAGMA_UNROLL
      for (int mm = 0; mm < MMA_M; ++mm) {
        const int r0 = row_of[2 * mm];
        const uint32_t packed = *reinterpret_cast<uint32_t const*>(
            &sMetaA((r0 >> 4) * 8 + (r0 & 7), k_block * 16 + lane_ai * 4, stage));
        pack_accum ^= packed;
        packedA_kb[mm] = packed;
      }
      return;
#endif
      const int kb = k_block * KB;
      CUTLASS_PRAGMA_UNROLL
      for (int mm = 0; mm < MMA_M; ++mm) {
        const uint32_t mA0 = load_meta16(row_of[2 * mm],     kb, stage);
        const uint32_t mA1 = load_meta16(row_of[2 * mm + 1], kb, stage);
        uint32_t packed = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int g = 0; g < 2; ++g) {
          packed |= (uint32_t)((mA0 >> (8 * g)) & 0x1F) << (5 * g);
          packed |= (uint32_t)((mA1 >> (8 * g)) & 0x1F) << (5 * (2 + g));
        }
        pack_accum ^= packed;
        packedA_kb[mm] = packed;
      }
    };
#endif

    int read_stage = smem_pipe_read.index();''')

rep('''    auto gemm_kblock = [&](auto k_block) {
      // (V,M) x (V,N) => (V,M,N)''', '''    auto gemm_kblock = [&](auto k_block) {
#if NVFP4_META_GROUPED && NVFP4_META_SW_CORRECT == 3 && NVFP4_META_TIE_OPERAND
      CUTLASS_PRAGMA_UNROLL
      for (int mm = 0; mm < MMA_M; ++mm) {
        CUTLASS_PRAGMA_UNROLL
        for (int nn = 0; nn < decltype(size<2>(accum))::value; ++nn) {
          asm volatile("lop3.b32 %0, %0, %1, %2, 0x78;" : "+f"(accum(0, mm, nn)) : "r"(packedA_kb[mm]), "r"(tie_zero));
        }
      }
#endif
      // (V,M) x (V,N) => (V,M,N)''')

PACK = '''#if NVFP4_META_GROUPED && NVFP4_META_SW_CORRECT == 3
        pack_kblock(k_block, read_stage);
#endif
'''
rep('''        auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

        if (k_block == K_BLOCK_MAX - 1) {''', '''        auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

''' + PACK + '''        if (k_block == K_BLOCK_MAX - 1) {''')
rep('''      auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

      if (k_block == K_BLOCK_MAX - 1) {''', '''      auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

''' + PACK.replace("        ", "      ") + '''      if (k_block == K_BLOCK_MAX - 1) {''')

rep('''      gemm_kblock(k_block);

    });
}''', '''      gemm_kblock(k_block);

    });
#if NVFP4_META_GROUPED && NVFP4_META_SW_CORRECT == 3
    if ((threadIdx.x & 31) == 0) g_nvfp4_pack_sink_grp[blockIdx.x % 2048] = pack_accum;
#endif
}''')

# ---- tensormaps ----------------------------------------------------------------------------
rep('''    cute::TmaDescriptor* tma_desc_sfb = &gmem_tensormap[sm_idx + 3 * sm_count];
''', '''    cute::TmaDescriptor* tma_desc_sfb = &gmem_tensormap[sm_idx + 3 * sm_count];
#if NVFP4_META_GROUPED
    cute::TmaDescriptor* tma_desc_metaa = &gmem_tensormap[sm_idx + 4 * sm_count];
#endif
''')
rep('''      copy(recast<uint128_t>(pSFB_tensormap), recast<uint128_t>(sSFB_tensormap));
    }
    __syncwarp();
    return cute::make_tuple(tma_desc_a, tma_desc_b, tma_desc_sfa, tma_desc_sfb);''', '''      copy(recast<uint128_t>(pSFB_tensormap), recast<uint128_t>(sSFB_tensormap));
#if NVFP4_META_GROUPED
      Tensor pMetaA_tensormap = make_tensor(mainloop_params.tma_load_metaa.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sMetaA_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_MetaA), Int<1>{}, Int<1>{});
      copy(recast<uint128_t>(pMetaA_tensormap), recast<uint128_t>(sMetaA_tensormap));
#endif
    }
    __syncwarp();
#if NVFP4_META_GROUPED
    return cute::make_tuple(tma_desc_a, tma_desc_b, tma_desc_sfa, tma_desc_sfb, tma_desc_metaa);
#else
    return cute::make_tuple(tma_desc_a, tma_desc_b, tma_desc_sfa, tma_desc_sfb);
#endif''')

rep('''    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_SFB,
                                                    mainloop_params.ptr_SFB[next_batch]);
''', '''    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_SFB,
                                                    mainloop_params.ptr_SFB[next_batch]);
#if NVFP4_META_GROUPED
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_MetaA,
                                                    mainloop_params.ptr_MetaA[next_batch]);
#endif
''')

rep('''    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_SFB,
                                                            prob_shape_SFB,
                                                            prob_stride_SFB);
''', '''    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_SFB,
                                                            prob_shape_SFB,
                                                            prob_stride_SFB);
#if NVFP4_META_GROUPED
    {
      cute::array<uint32_t, MaxTensorRank> prob_shape_Meta  = {1,1,1,1,1};
      cute::array<uint64_t, MaxTensorRank> prob_stride_Meta = {0,0,0,0,0};
      Tensor tensor_meta = make_tensor(static_cast<ElementMeta const*>(nullptr),
          make_layout(make_shape(int32_t(M / MetaRD), int32_t(K / MetaCD), Int<1>{}),
                      meta_stride(int64_t(M / MetaRD), int64_t(K / MetaCD))));
      cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_metaa, tensor_meta,
                                               prob_shape_Meta, prob_stride_Meta);
      // uint8 elements: element strides are already byte strides
      cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_MetaA,
                                                              prob_shape_Meta,
                                                              prob_stride_Meta);
    }
#endif
''')

# generic tuple signatures for the tensormap helpers
rep('''  template <class TensorMapA, class TensorMapB, class TensorMapSFA, class TensorMapSFB, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_perform_update(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      cute::tuple<TensorMapA, TensorMapB, TensorMapSFA, TensorMapSFB> const& input_tensormaps,''', '''  template <class InputTensorMaps, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_perform_update(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      InputTensorMaps const& input_tensormaps,''')
rep('''  template <class TensorMapA, class TensorMapB, class TensorMapSFA, class TensorMapSFB>
  CUTLASS_DEVICE
  void
  tensormaps_cp_fence_release (
      TensorMapStorage& shared_tensormaps,
      cute::tuple<TensorMapA, TensorMapB, TensorMapSFA, TensorMapSFB> const& input_tensormaps) {''', '''  template <class InputTensorMaps>
  CUTLASS_DEVICE
  void
  tensormaps_cp_fence_release (
      TensorMapStorage& shared_tensormaps,
      InputTensorMaps const& input_tensormaps) {''')
rep('''    tma_descriptor_cp_fence_release(get<3>(input_tensormaps), shared_tensormaps.smem_tensormap_SFB);
''', '''    tma_descriptor_cp_fence_release(get<3>(input_tensormaps), shared_tensormaps.smem_tensormap_SFB);
#if NVFP4_META_GROUPED
    tma_descriptor_cp_fence_release(get<4>(input_tensormaps), shared_tensormaps.smem_tensormap_MetaA);
#endif
''')
rep('''  template <class TensorMapA, class TensorMapB, class TensorMapSFA, class TensorMapSFB>
  CUTLASS_DEVICE
  void
  tensormaps_fence_acquire(cute::tuple<TensorMapA, TensorMapB, TensorMapSFA, TensorMapSFB> const& input_tensormaps) {''', '''  template <class InputTensorMaps>
  CUTLASS_DEVICE
  void
  tensormaps_fence_acquire(InputTensorMaps const& input_tensormaps) {''')
rep('''    cute::tma_descriptor_fence_acquire(get<3>(input_tensormaps));
''', '''    cute::tma_descriptor_fence_acquire(get<3>(input_tensormaps));
#if NVFP4_META_GROUPED
    cute::tma_descriptor_fence_acquire(get<4>(input_tensormaps));
#endif
''')

open(sys.argv[2], "w").write(src)
print("patched", sys.argv[2])
