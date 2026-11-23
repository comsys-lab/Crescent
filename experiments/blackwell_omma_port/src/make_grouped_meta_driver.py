#!/usr/bin/env python3
"""Builds src/grouped_meta.cu from CUTLASS 79d_blackwell_geforce_nvfp4_grouped_gemm.cu:
 * NVFP4_META_GROUPED=1: per-group correction-data buffer (M_g x K_g/GS bytes) passed as
   mainloop ptr_MetaA (patched collective, patches/patch_grouped_meta.py);
 * NVFP4_SMEM_PAD extra epilogue carveout (same Stage count for baseline / Czardas builds);
 * prints the mainloop Stage count;
 * timing: the example's loop (initialize + run per iteration, as in grouped_raw.txt) and a
   kernel-only loop (initialize once, run per iteration).
Usage: make_grouped_meta_driver.py <79d.cu> <out.cu>"""
import sys
s = open(sys.argv[1]).read()
def rep(o, n, c=1):
    global s
    assert s.count(o) == c, (s.count(o), o[:70]); s = s.replace(o, n)

rep('''#include "helper.h"
''', '''#include "helper.h"

#ifndef NVFP4_SMEM_PAD
#define NVFP4_SMEM_PAD 0
#endif
#ifndef NVFP4_META_GROUPED
#define NVFP4_META_GROUPED 0
#endif
#ifndef NVFP4_META_GROUP_SIZE
#define NVFP4_META_GROUP_SIZE 8
#endif
''')
rep('''  static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,''',
    '''  static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage)) + NVFP4_SMEM_PAD>,''', 2)
rep('''cutlass::DeviceAllocation<ElementAccumulator> norm_constant_device;
''', '''cutlass::DeviceAllocation<ElementAccumulator> norm_constant_device;
#if NVFP4_META_GROUPED
std::vector<cutlass::DeviceAllocation<uint8_t>> block_MetaA;
cutlass::DeviceAllocation<const uint8_t *> ptr_MetaA;
#endif
''')
rep('''  norm_constant_device.reset(1);
  norm_constant_device.copy_from_host(&options.norm_constant);
}''', '''  norm_constant_device.reset(1);
  norm_constant_device.copy_from_host(&options.norm_constant);
#if NVFP4_META_GROUPED
  // correction data: 1 byte per GS-element K-group of A, (M, K/GS) row-major per group
  std::vector<const uint8_t *> ptr_MetaA_host(options.groups);
  block_MetaA.resize(options.groups);
  for (int32_t i = 0; i < options.groups; ++i) {
    auto problem = options.problem_sizes_host.at(i);
    size_t bytes = size_t(get<0>(problem)) * (get<2>(problem) / NVFP4_META_GROUP_SIZE);
    std::vector<uint8_t> h(bytes);
    for (size_t j = 0; j < bytes; ++j) h[j] = uint8_t(rand());
    block_MetaA[i].reset(bytes);
    block_MetaA[i].copy_from_host(h.data());
    ptr_MetaA_host[i] = block_MetaA[i].get();
  }
  ptr_MetaA.reset(options.groups);
  ptr_MetaA.copy_from_host(ptr_MetaA_host.data());
#endif
}''')
rep('''       ptr_SFA.get(), layout_SFA.get(), ptr_SFB.get(), layout_SFB.get()},''',
    '''       ptr_SFA.get(), layout_SFA.get(), ptr_SFB.get(), layout_SFB.get()
#if NVFP4_META_GROUPED
       , ptr_MetaA.get()
#endif
      },''', 2)
rep('''  std::cout << "  Groups      : " << options.groups  << std::endl;
''', '''  std::cout << "  Groups      : " << options.groups  << std::endl;
  std::cout << "  Stages      : " << Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Stages
            << "  (meta=" << NVFP4_META_GROUPED << ", pad=" << NVFP4_SMEM_PAD << ")" << std::endl;
''')
rep('''    std::cout << "  TFLOPS      : " << result.gflops / 1000.0 << std::endl;
''', '''    std::cout << "  TFLOPS      : " << result.gflops / 1000.0 << std::endl;

    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
    GpuTimer timer2;
    timer2.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.run(/* stream = */ nullptr, /* cuda_adapter = */ nullptr, /* launch_with_pdl = */ options.use_pdl));
    }
    timer2.stop();
    std::cout << "  Kernel-only avg runtime : " << double(timer2.elapsed_millis()) / double(options.iterations) << " ms" << std::endl;
''')
open(sys.argv[2], "w").write(s)
