import os, sys, torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM

ATTN_KEYS = ("flash", "fmha", "attention", "mha", "sdpa", "softmax", "scaled_dot")
def is_attn(name):
    return any(k in name.lower() for k in ATTN_KEYS)

MS = [int(a) for a in sys.argv[1:]] or [2048, 4096, 8192]
NREP = 5
model = AutoModelForCausalLM.from_pretrained(os.environ.get("LLAMA_MODEL", "meta-llama/Llama-3.1-8B"), dtype=torch.bfloat16,
                                             attn_implementation="sdpa", device_map={"": 0}).eval()
kw = dict(use_cache=False, logits_to_keep=1)
results = {M: [] for M in MS}
for rep in range(NREP):
    for M in MS:
        ids = torch.randint(0, 30000, (1, M), device="cuda")
        with torch.inference_mode():
            for _ in range(2):
                model(ids, **kw)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(2):
                    model(ids, **kw)
                torch.cuda.synchronize()
        attn_ms = 0.0
        for e in prof.key_averages():
            us = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
            if us <= 0: continue
            if is_attn(e.key):
                attn_ms += us / 1e3 / 2.0
        results[M].append(attn_ms)
        print(f"rep{rep} M={M:>6}  real_fused_attn_bf16={attn_ms:8.2f}ms", flush=True)

print(f"\nSUMMARY (avg of {NREP} runs)")
for M in MS:
    r = results[M]
    avg = sum(r) / len(r)
    print(f"M={M:>6}  avg={avg:8.2f}  range={min(r):.2f}..{max(r):.2f}  reps={['%.2f'%x for x in r]}")
