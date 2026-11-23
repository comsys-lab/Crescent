import ctypes, sys, torch, torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM

lib_t = ctypes.CDLL("/work/libnvfp4_truebase.so")
lib_s = ctypes.CDLL("/work/libnvfp4_sel.so")
for lib in (lib_t, lib_s):
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

ATTN_KEYS = ("flash", "fmha", "attention", "mha", "sdpa", "softmax", "scaled_dot")
GEMM_KEYS = ("gemm", "cutlass", "cublas", "quantize_nvfp4", "nvfp4")
def bucket(name):
    n = name.lower()
    if any(k in n for k in ATTN_KEYS): return "attention"
    if any(k in n for k in GEMM_KEYS): return "linear_gemm"
    return "other"

MS = [int(a) for a in sys.argv[1:]] or [2048, 4096, 8192]
NREP = 5
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16, device_map={"": 0}).eval()
lib_t.nvfp4_init(max(MS), 14336); lib_s.nvfp4_init(max(MS), 14336)
for lib in (lib_t, lib_s):
    lib.nvfp4_set_act_scale(0.05)

orig = {}
true_lins, sel_lins = {}, {}
for L in model.model.layers:
    for parent, names in ((L.self_attn, ["q_proj", "k_proj", "v_proj", "o_proj"]), (L.mlp, ["gate_proj", "up_proj", "down_proj"])):
        for n in names:
            lin = getattr(parent, n)
            key = (id(parent), n)
            orig[key] = (parent, n)
            true_lins[key] = NV(lin, lib_t)
            sel_lins[key] = NV(lin, lib_s)
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

def gemm_only_ms(ids):
    kw = dict(use_cache=False, logits_to_keep=1)
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(2):
            model(ids, **kw)
        torch.cuda.synchronize()
    tot = {}
    for e in prof.key_averages():
        us = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
        if us <= 0: continue
        tot[bucket(e.key)] = tot.get(bucket(e.key), 0.0) + us / 1e3 / 2.0
    return tot.get("linear_gemm", 0.0)

ids = {M: torch.randint(0, 30000, (1, M), device="cuda") for M in MS}
results = {M: {"true_fwd": [], "sel_fwd": [], "true_gemm": [], "sel_gemm": []} for M in MS}
for rep in range(NREP):
    for M in MS:
        install(true_lins)
        tf = timeit(ids[M]); tg = gemm_only_ms(ids[M])
        install(sel_lins)
        sf = timeit(ids[M]); sg = gemm_only_ms(ids[M])
        results[M]["true_fwd"].append(tf); results[M]["sel_fwd"].append(sf)
        results[M]["true_gemm"].append(tg); results[M]["sel_gemm"].append(sg)
        print(f"rep{rep} M={M:>6}  fwd: true={tf:8.2f} sel={sf:8.2f}   gemm: true={tg:7.2f} sel={sg:7.2f}", flush=True)

print(f"\nSUMMARY (avg of {NREP} runs, +/- is min..max range)")
for M in MS:
    r = results[M]
    tf_avg = sum(r["true_fwd"]) / NREP; sf_avg = sum(r["sel_fwd"]) / NREP
    tg_avg = sum(r["true_gemm"]) / NREP; sg_avg = sum(r["sel_gemm"]) / NREP
    fwd_delta = [s - t for s, t in zip(r["sel_fwd"], r["true_fwd"])]
    gemm_delta = [s - t for s, t in zip(r["sel_gemm"], r["true_gemm"])]
    print(f"M={M:>6}")
    print(f"  FULL FWD : true_avg={tf_avg:8.2f} sel_avg={sf_avg:8.2f} delta_avg={sf_avg-tf_avg:7.2f} "
          f"(range {min(fwd_delta):.2f}..{max(fwd_delta):.2f})  pct={100*(sf_avg-tf_avg)/tf_avg:.2f}%")
    print(f"  GEMM ONLY: true_avg={tg_avg:8.2f} sel_avg={sg_avg:8.2f} delta_avg={sg_avg-tg_avg:7.2f} "
          f"(range {min(gemm_delta):.2f}..{max(gemm_delta):.2f})  pct={100*(sg_avg-tg_avg)/tg_avg:.2f}%")
