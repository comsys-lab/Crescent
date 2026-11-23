import ctypes, sys, torch, torch.nn as nn
from transformers import AutoModelForCausalLM
lib = ctypes.CDLL(__import__("os").environ.get("LIB","/work/code/e2e/libnvfp4.so"))
lib.nvfp4_prepare.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int]; lib.nvfp4_prepare.restype=ctypes.c_int
lib.nvfp4_linear.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]; lib.nvfp4_linear.restype=ctypes.c_int
lib.nvfp4_set_act_scale.argtypes=[ctypes.c_float]
DYN = 0
class NV(nn.Module):
    def __init__(s, lin):
        super().__init__()
        s.N, s.K = lin.weight.shape
        s.id = lib.nvfp4_prepare(lin.weight.data_ptr(), s.N, s.K); torch.cuda.synchronize()
    def forward(s, x):
        sh = x.shape; x2 = x.reshape(-1, s.K).contiguous(); M = x2.shape[0]
        out = torch.empty(M, s.N, dtype=torch.bfloat16, device=x.device)
        r = lib.nvfp4_linear(s.id, x2.data_ptr(), M, out.data_ptr(), torch.cuda.current_stream().cuda_stream, DYN)
        assert r == 0, r
        return out.reshape(*sh[:-1], s.N)
MS = [int(a) for a in sys.argv[1:]] or [2048, 4096, 8192, 16384]
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16, device_map={"":0}).eval()
def timeit(ids, n=8):
    kw = dict(use_cache=False, logits_to_keep=1)
    with torch.inference_mode():
        model(ids, **kw); torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
        for _ in range(n): model(ids, **kw)
        b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n
ids = {M: torch.randint(0, 30000, (1, M), device="cuda") for M in MS}
bf16 = {M: timeit(ids[M]) for M in MS}
lib.nvfp4_init(max(MS), 14336)
# correctness on one layer with a calibrated activation scale
lin = model.model.layers[0].mlp.down_proj; x = torch.randn(256, 14336, device="cuda", dtype=torch.bfloat16)
ref = lin(x).float(); nv = NV(lin); lib.nvfp4_set_act_scale(float(x.abs().max())/(448*6))
o = nv(x).float(); print("layer check rel err: %.3f%%" % (100*(o-ref).norm()/ref.norm()), flush=True)
swapped = 0
for L in model.model.layers:
    for parent, names in ((L.self_attn, ["q_proj","k_proj","v_proj","o_proj"]), (L.mlp, ["gate_proj","up_proj","down_proj"])):
        for n in names: setattr(parent, n, NV(getattr(parent, n))); swapped += 1
torch.cuda.empty_cache(); print("swapped", swapped, flush=True)
lib.nvfp4_set_act_scale(0.05)
res = {}
for dyn in (0, 1):
    DYN = dyn
    for M in MS: res[(dyn, M)] = timeit(ids[M])
print("M, bf16_fwd_ms, nvfp4_fwd_static_ms, nvfp4_fwd_dynamic_amax_ms")
for M in MS: print(M, round(bf16[M],1), round(res[(0,M)],1), round(res[(1,M)],1))
