import ctypes, sys, os, csv, torch, torch.nn as nn
from transformers import AutoModelForCausalLM
from torch.profiler import profile, ProfilerActivity

lib = ctypes.CDLL(os.environ.get("LIB", "/work/code/libnvfp4.so"))
lib.nvfp4_prepare.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]; lib.nvfp4_prepare.restype = ctypes.c_int
lib.nvfp4_linear.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]; lib.nvfp4_linear.restype = ctypes.c_int
lib.nvfp4_set_act_scale.argtypes = [ctypes.c_float]

class NV(nn.Module):
    def __init__(s, lin):
        super().__init__()
        s.N, s.K = lin.weight.shape
        s.id = lib.nvfp4_prepare(lin.weight.data_ptr(), s.N, s.K); torch.cuda.synchronize()
    def forward(s, x):
        sh = x.shape; x2 = x.reshape(-1, s.K).contiguous(); M = x2.shape[0]
        out = torch.empty(M, s.N, dtype=torch.bfloat16, device=x.device)
        r = lib.nvfp4_linear(s.id, x2.data_ptr(), M, out.data_ptr(), torch.cuda.current_stream().cuda_stream, 0)
        assert r == 0, r
        return out.reshape(*sh[:-1], s.N)

GEMM_KEYS = ("gemm", "cutlass", "cublas", "sgemm", "hgemm", "ampere", "nvjet", "s16816",
             "h16816", "wgrad", "dgrad", "gemv", "splitk", "sm80", "sm90", "16x8", "8x8",
             "quantize_nvfp4", "nvfp4")
ATTN_KEYS = ("flash", "fmha", "attention", "mha", "sdpa", "softmax", "scaled_dot")
EW_KEYS = ("norm", "rope", "rotary", "silu", "gelu", "swiglu", "elementwise", "relu",
           "mul", "add", "cast", "copy", "layer_norm", "rms", "activation", "dropout",
           "index", "embedding", "reduce", "fill", "arange", "triu", "where", "bias")

def bucket(name):
    n = name.lower()
    if any(k in n for k in ATTN_KEYS): return "attention"
    if any(k in n for k in GEMM_KEYS): return "linear_gemm"
    if any(k in n for k in EW_KEYS): return "norm_rope_elementwise"
    return "other"

MS = [int(a) for a in sys.argv[1:]] or [2048, 4096, 8192]
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16, device_map={"": 0}).eval()

lib.nvfp4_init(max(MS), 14336)
lin = model.model.layers[0].mlp.down_proj
x = torch.randn(256, 14336, device="cuda", dtype=torch.bfloat16)
ref = lin(x).float(); nv = NV(lin); lib.nvfp4_set_act_scale(float(x.abs().max()) / (448 * 6))
o = nv(x).float()
print("layer check rel err: %.3f%%" % (100 * (o - ref).norm() / ref.norm()), flush=True)

swapped = 0
for L in model.model.layers:
    for parent, names in ((L.self_attn, ["q_proj", "k_proj", "v_proj", "o_proj"]), (L.mlp, ["gate_proj", "up_proj", "down_proj"])):
        for n in names:
            setattr(parent, n, NV(getattr(parent, n))); swapped += 1
torch.cuda.empty_cache()
print("swapped", swapped, flush=True)
lib.nvfp4_set_act_scale(0.05)

OUT = os.environ.get("OUT_DIR", "/work/results/kernels")
os.makedirs(OUT, exist_ok=True)
kw = dict(use_cache=False, logits_to_keep=1)
for M in MS:
    ids = torch.randint(0, 30000, (1, M), device="cuda")
    with torch.inference_mode():
        for _ in range(2):
            model(ids, **kw)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
        for _ in range(4):
            model(ids, **kw)
        b.record(); torch.cuda.synchronize()
        fwd_ms = a.elapsed_time(b) / 4

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(2):
                model(ids, **kw)
            torch.cuda.synchronize()

    krows = []
    for e in prof.key_averages():
        us = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
        if us <= 0:
            continue
        krows.append((us / 1e3 / 2.0, e.count / 2.0, bucket(e.key), e.key))
    krows.sort(reverse=True)
    kcsv = os.path.join(OUT, f"Llama-3.1-8B_NVFP4baseline_M{M}.csv")
    with open(kcsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ms_per_forward", "launches_per_forward", "category", "kernel"])
        for ms, c, cat, name in krows:
            w.writerow([f"{ms:.5f}", f"{c:.1f}", cat, name])
    cats = {}
    for ms, c, cat, name in krows:
        cats[cat] = cats.get(cat, 0.0) + ms
    gpu_sum = sum(cats.values())
    print(f"M={M:>6}  fwd_ms(event)={fwd_ms:8.2f}  gpu_sum(profiler)={gpu_sum:8.2f}  " +
          "  ".join(f"{k}={v:.2f}" for k, v in sorted(cats.items(), key=lambda x: -x[1])), flush=True)
