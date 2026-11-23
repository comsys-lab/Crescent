import ctypes, os, torch
lib = ctypes.CDLL(os.environ["LIB"])
lib.nvfp4_init.argtypes=[ctypes.c_int,ctypes.c_int]
lib.nvfp4_prepare.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int]; lib.nvfp4_prepare.restype=ctypes.c_int
lib.nvfp4_linear.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]; lib.nvfp4_linear.restype=ctypes.c_int
lib.nvfp4_set_act_scale.argtypes=[ctypes.c_float]
lib.nvfp4_init(2048, 14336)
W = torch.randn(4096,4096, device="cuda", dtype=torch.bfloat16)
lid = lib.nvfp4_prepare(W.data_ptr(), 4096, 4096)
x = torch.randn(2048,4096, device="cuda", dtype=torch.bfloat16)
out = torch.empty(2048,4096, device="cuda", dtype=torch.bfloat16)
lib.nvfp4_set_act_scale(0.05)
r = lib.nvfp4_linear(lid, x.data_ptr(), 2048, out.data_ptr(), torch.cuda.current_stream().cuda_stream, 0)
torch.cuda.synchronize()
print("ran, r=",r)
