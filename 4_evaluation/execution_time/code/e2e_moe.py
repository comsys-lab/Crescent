import ctypes, os, sys, gc, inspect, torch, torch.nn as nn
lib = ctypes.CDLL(os.environ["LIB"])
lib.nvfp4_prepare.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int]; lib.nvfp4_prepare.restype=ctypes.c_int
lib.nvfp4_linear.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]; lib.nvfp4_linear.restype=ctypes.c_int
lib.nvfp4_init.argtypes=[ctypes.c_int,ctypes.c_int]
MODEL, MS = sys.argv[1], [int(a) for a in sys.argv[2:]]
ROLES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
class NV(nn.Module):
    def __init__(s, lin, ident):
        super().__init__(); s.N, s.K = lin.weight.shape; s.id = ident
    def forward(s, x):
        sh = x.shape; x2 = x.reshape(-1, s.K).contiguous(); M = x2.shape[0]
        out = torch.empty(M, s.N, dtype=torch.bfloat16, device=x.device)
        r = lib.nvfp4_linear(s.id, x2.data_ptr(), M, out.data_ptr(), torch.cuda.current_stream(x.device).cuda_stream, 1)
        assert r == 0, (r, M, s.N, s.K)
        return out.reshape(*sh[:-1], s.N)
ng = torch.cuda.device_count()
if MODEL == "qwen":
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", dtype=torch.bfloat16, device_map="auto", max_memory={i:"20GiB" for i in range(ng)}).eval()
else:
    sys.path.insert(0, "/hyp_common"); import deepseek_moe_model as dsm
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from huggingface_hub import snapshot_download
    d = snapshot_download(dsm.MODEL_NAME, allow_patterns=["*.safetensors","*.json"]); cfg = dsm.load_config(d)
    with init_empty_weights(): model = dsm.DeepseekMoEForCausalLM(cfg)
    model = load_checkpoint_and_dispatch(model, d, device_map="auto", dtype=torch.bfloat16, max_memory={i:"10GiB" for i in range(ng)}, no_split_module_classes=["DecoderLayer"]).eval()
first = next(model.parameters()).device
swapped = kept = 0; inited = set()
targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and any(n.endswith(r) for r in ROLES)]
for name, lin in targets:
    dev = lin.weight.device
    with torch.cuda.device(dev):
        if dev.index not in inited: lib.nvfp4_init(max(MS), 14336); inited.add(dev.index)
        N, K = lin.weight.shape
        ok = (K % 128 == 0) and (N % 128 == 0)
        if ok:
            ident = lib.nvfp4_prepare(lin.weight.data_ptr(), N, K); torch.cuda.synchronize()
            z = torch.zeros(128, K, dtype=torch.bfloat16, device=dev); o = torch.empty(128, N, dtype=torch.bfloat16, device=dev)
            ok = lib.nvfp4_linear(ident, z.data_ptr(), 128, o.data_ptr(), torch.cuda.current_stream(dev).cuda_stream, 1) == 0
    if not ok: kept += 1; print("kept bf16:", name, tuple(lin.weight.shape)); continue
    parent = model.get_submodule(name.rsplit(".",1)[0]) if "." in name else model
    setattr(parent, name.rsplit(".",1)[-1], NV(lin, ident)); swapped += 1
del targets; gc.collect()
for i in range(ng):
    with torch.cuda.device(i): torch.cuda.empty_cache()
print(f"swapped {swapped} linears, kept bf16 {kept}", flush=True)
kw = dict(use_cache=False)
if "logits_to_keep" in inspect.signature(model.forward).parameters: kw["logits_to_keep"] = 1
from torch.profiler import profile, ProfilerActivity
def timeit(ids, n=2):
    with torch.inference_mode():
        model(ids, **kw); torch.cuda.synchronize()
        res = []
        for _ in range(n):
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                model(ids, **kw); torch.cuda.synchronize()
            res.append(sum(e.self_device_time_total for e in prof.key_averages()) / 1000.0)
        return min(res)   # summed GPU kernel time over all devices (ms); host launch gaps excluded
print("M, nvfp4_gpu_kernel_ms_sum")
for M in MS:
    torch.manual_seed(0); ids = torch.randint(0, 30000, (1, M), device=first)
    try: print(M, round(timeit(ids), 1), flush=True)
    except torch.OutOfMemoryError: print(M, "OOM", flush=True)
