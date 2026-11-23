"""Non-GEMM GPU kernel time of a Qwen3-30B-A3B / DeepSeek-MoE-16B prefill forward with a vectorized
(non-per-expert-loop) MoE dispatch, plus the real fused BF16 attention (SDPA) time.

The elementwise ops are written exactly as HF eager does them (same kernels as in the Llama
denominator: RMSNorm as fp32 pow/mean/rsqrt/mul, RoPE as cat-based rotate_half, SiLU then mul);
only HF's per-expert Python loop is replaced by one sort-based permute / index_add unpermute, i.e.
what a grouped-GEMM MoE needs. GEMMs (all Linear layers, QK^T, PV) are excluded: they are measured
separately (NVFP4 / grouped GEMM). The router gate stays a BF16 Linear (kept BF16 in e2e_moe.py too).

Kernel time = summed CUDA self time from torch.profiler (same metric as e2e_moe.py), per layer,
averaged over R repetitions; NREP outer reps reported.
Usage: python measure_moe_other.py  -> prints 'rep<r> <model> M=<M> layer_nonmoe= layer_moe= attn_sdpa= fwd_once= dense0_act=' (ms)
"""
import torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

dev = "cuda"; bf = torch.bfloat16
ATTN_KEYS = ("flash", "fmha", "attention", "mha", "sdpa", "softmax", "scaled_dot")
CFG = {
    "qwen": dict(H=2048, nh=32, nkv=4, hd=128, E=128, topk=8, I=768, qknorm=True, norm_topk=True,
                 shared=0, vocab=151936, dense_I=0),
    "ds":   dict(H=2048, nh=16, nkv=16, hd=128, E=64, topk=6, I=1408, qknorm=False, norm_topk=False,
                 shared=2816, vocab=102400, dense_I=10944),
}
R, NREP = 5, 3


def rmsnorm(x, w, eps=1e-6):
    h = x.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    return w * h.to(x.dtype)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rope(q, k, cos, sin):
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def kernel_ms(fn, n):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    tot = attn = 0.0
    for e in prof.key_averages():
        us = getattr(e, "self_device_time_total", 0) or 0
        tot += us
        if any(k in e.key.lower() for k in ATTN_KEYS): attn += us
    return tot / 1e3 / n, attn / 1e3 / n


@torch.inference_mode()
def run(name, M):
    c = CFG[name]; H, nh, nkv, hd, E, K = c["H"], c["nh"], c["nkv"], c["hd"], c["E"], c["topk"]
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(1, M, H, device=dev, dtype=bf, generator=g)
    w = torch.ones(H, device=dev, dtype=bf); wh = torch.ones(hd, device=dev, dtype=bf)
    q = torch.randn(1, M, nh * hd, device=dev, dtype=bf, generator=g)          # q/k/v/o projection outputs
    k = torch.randn(1, M, nkv * hd, device=dev, dtype=bf, generator=g)
    v = torch.randn(1, M, nkv * hd, device=dev, dtype=bf, generator=g)
    ao = torch.randn(1, nh, M, hd, device=dev, dtype=bf, generator=g)          # attention output
    o = torch.randn(1, M, H, device=dev, dtype=bf, generator=g)                 # o_proj / MoE outputs
    cos = torch.randn(1, 1, M, hd, device=dev, dtype=bf, generator=g); sin = cos.clone()
    Wr = torch.randn(E, H, device=dev, dtype=bf, generator=g) * 0.02           # router (BF16 Linear)
    gu = torch.randn(M * K, c["I"], device=dev, dtype=bf, generator=g)         # grouped gate / up outputs
    dn = torch.randn(M * K, H, device=dev, dtype=bf, generator=g)              # grouped down output
    if c["shared"]:
        sg = torch.randn(M, c["shared"], device=dev, dtype=bf, generator=g)

    def layer_attn_part():
        h = rmsnorm(x, w)                                                       # input_layernorm
        qq = q.view(1, M, nh, hd); kk = k.view(1, M, nkv, hd)
        if c["qknorm"]:
            qq, kk = rmsnorm(qq, wh), rmsnorm(kk, wh)
        qq, kk = qq.transpose(1, 2), kk.transpose(1, 2)
        qq, kk = rope(qq, kk, cos, sin)
        a = ao.transpose(1, 2).reshape(1, M, nh * hd).contiguous()              # -> o_proj input
        r = x + o                                                               # residual
        h2 = rmsnorm(r, w)                                                      # post_attention_layernorm
        r2 = r + o                                                              # residual after MLP
        return a, h2, r2

    h2 = rmsnorm(x, w)
    def layer_moe_part():
        # --- MoE routing, vectorized ---
        logits = F.linear(h2.view(M, H), Wr)
        p = logits.softmax(dim=-1, dtype=torch.float)
        tw, ti = torch.topk(p, K, dim=-1)
        if c["norm_topk"]:
            tw = tw / tw.sum(dim=-1, keepdim=True)
        tw = tw.to(bf)
        flat = ti.view(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=E)                              # grouped-GEMM problem sizes
        tok = order // K
        xp = h2.view(M, H).index_select(0, tok)                                 # permute
        act = F.silu(gu) * gu                                                   # act(gate) * up
        y = dn * tw.view(-1)[order].unsqueeze(-1)
        out = torch.zeros(M, H, device=dev, dtype=bf).index_add_(0, tok, y)     # unpermute
        if c["shared"]:
            sa = F.silu(sg) * sg                                                # shared expert act
            out = out + o.view(M, H)                                            # + shared expert output
        return xp, act, counts, out

    def attn():
        qq = q.view(1, M, nh, hd).transpose(1, 2); kk = k.view(1, M, nkv, hd).transpose(1, 2)
        vv = v.view(1, M, nkv, hd).transpose(1, 2)
        return F.scaled_dot_product_attention(qq, kk, vv, is_causal=True, enable_gqa=(nkv != nh))

    def once():                                                                 # per forward
        hl = rmsnorm(x[:, -1:], w)                                              # final norm, last token
        return F.linear(hl, torch.empty(c["vocab"], H, device=dev, dtype=bf))   # lm_head, logits_to_keep=1

    dense_ms = 0.0
    if c["dense_I"]:
        di = torch.randn(M, c["dense_I"], device=dev, dtype=bf, generator=g)
        dense_ms = kernel_ms(lambda: F.silu(di) * di, R)[0]                     # layer-0 dense MLP act
    la, _ = kernel_ms(layer_attn_part, R)
    lm, _ = kernel_ms(layer_moe_part, R)
    _, at = kernel_ms(attn, R)
    on, _ = kernel_ms(once, R)
    return la, lm, at, on, dense_ms


for rep in range(1, NREP + 1):
    for name in ("qwen", "ds"):
        for M in (2048, 4096, 8192):
            la, lm, at, on, dm = run(name, M)
            print(f"rep{rep} {name} M={M} layer_nonmoe={la:.4f} layer_moe={lm:.4f} attn_sdpa={at:.4f} fwd_once={on:.4f} dense0_act={dm:.4f}", flush=True)
            torch.cuda.empty_cache()
