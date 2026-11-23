import ctypes, sys, torch, torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM

# Both libraries use the SAME true-base (unpatched) GEMM -- the ONLY difference is whether the
# quantize kernel also runs the real selection search + correction-data computation (discarded,
# no metadata write, no metaA). Isolates the real end-to-end cost of the SELECTION COMPUTATION
# alone, with zero change to data movement -- answers the "<0.5%" claim directly via a true
# with/without full-model run, instead of the weighted microbenchmark-assembly estimate.
lib_t = ctypes.CDLL("/work/libnvfp4_truebase_new.so")
lib_c = ctypes.CDLL("/work/libnvfp4_computeonly.so")
for lib in (lib_t, lib_c):
    lib.nvfp4_prepare.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]; lib.nvfp4_prepare.restype = ctypes.c_int
    lib.nvfp4_linear.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]; lib.nvfp4_linear.restype = ctypes.c_int
    lib.nvfp4_set_act_scale.argtypes = [ctypes.c_float]
    lib.nvfp4_init.argtypes = [ctypes.c_int, ctypes.c_int]

class NV(nn.Module):
    def __init__(s, lin, lib):
        super().__init__()
        s.lib = lib; s.N, s.K = lin.weight.shape
        s.id = lib.nvfp4_prepare(lin.weight.data_ptr(), s.N, s.K); torch.cuda.synchronize()
    def forward(s, x):
        sh = x.shape; x2 = x.reshape(-1, s.K).contiguous(); M = x2.shape[0]
        out = torch.empty(M, s.N, dtype=torch.bfloat16, device=x.device)
        r = s.lib.nvfp4_linear(s.id, x2.data_ptr(), M, out.data_ptr(), torch.cuda.current_stream().cuda_stream, 0)
        assert r == 0, r
        return out.reshape(*sh[:-1], s.N)

MS = [int(a) for a in sys.argv[1:]] or [2048, 4096, 8192]
NREP = 5
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16, device_map={"": 0}).eval()
lib_t.nvfp4_init(max(MS), 14336); lib_c.nvfp4_init(max(MS), 14336)
for lib in (lib_t, lib_c):
    lib.nvfp4_set_act_scale(0.05)

orig = {}
true_lins, co_lins = {}, {}
for L in model.model.layers:
    for parent, names in ((L.self_attn, ["q_proj", "k_proj", "v_proj", "o_proj"]), (L.mlp, ["gate_proj", "up_proj", "down_proj"])):
        for n in names:
            lin = getattr(parent, n)
            key = (id(parent), n)
            orig[key] = (parent, n)
            true_lins[key] = NV(lin, lib_t)
            co_lins[key] = NV(lin, lib_c)
            setattr(parent, n, None); del lin
torch.cuda.empty_cache()
print(f"built {len(orig)} layers for each variant", flush=True)

def install(variant_dict):
    for key, (parent, n) in orig.items():
        setattr(parent, n, variant_dict[key])

def timeit(ids, n=8):
    kw = dict(use_cache=False, logits_to_keep=1)
    with torch.inference_mode():
        model(ids, **kw); torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
        for _ in range(n):
            model(ids, **kw)
        b.record(); torch.cuda.synchronize()
        return a.elapsed_time(b) / n

ids = {M: torch.randint(0, 30000, (1, M), device="cuda") for M in MS}
results = {M: {"true_fwd": [], "co_fwd": []} for M in MS}
for rep in range(NREP):
    for M in MS:
        install(true_lins)
        tf = timeit(ids[M])
        install(co_lins)
        cf = timeit(ids[M])
        results[M]["true_fwd"].append(tf); results[M]["co_fwd"].append(cf)
        print(f"rep{rep} M={M:>6}  fwd: true={tf:8.2f} computeonly={cf:8.2f}", flush=True)

print(f"\nSUMMARY (avg of {NREP} runs, +/- is min..max range)")
for M in MS:
    r = results[M]
    tf_avg = sum(r["true_fwd"]) / NREP; cf_avg = sum(r["co_fwd"]) / NREP
    fwd_delta = [c - t for c, t in zip(r["co_fwd"], r["true_fwd"])]
    print(f"M={M:>6}")
    print(f"  FULL FWD : true_avg={tf_avg:8.2f} computeonly_avg={cf_avg:8.2f} delta_avg={cf_avg-tf_avg:7.2f} "
          f"(range {min(fwd_delta):.2f}..{max(fwd_delta):.2f})  pct={100*(cf_avg-tf_avg)/tf_avg:.2f}%")
