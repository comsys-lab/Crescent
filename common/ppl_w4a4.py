"""Model builder and PPL loop shared by Table 1, Table 6 and Table 7.

Settings (arms), as in the paper's Tables 6 and 7:
  bf16      : BF16 weights and activations.
  native    : GPTQ 4-bit weights; every activation operand feeding a matmul (Linear inputs, Q, K, V, P)
              in the original E2M1 format (NVFP4: block 16, E4M3 scale; MXFP4: block 32, E8M0 scale).
  magnitude : as native, but in every 16-element group the elements with the largest |a| take their E2M3
              value under the same block scale: Linear inputs top-2, Q/K/V/P top-1.
  crescent  : as magnitude, but the elements are selected by squared quantization error.
  all_e2m3  : as native, but every activation element in E2M3.
QK^T and P@V use the decomposed form for magnitude/crescent (selected x selected delta term dropped).
P is quantized on causal positions only. Weights for every quantized arm come from the GPTQ cache built in
weight_quantization/ ($PTQ_CACHE_ROOT/{fmt}/{model}/ptq_cache_NVFP4.pt).

Models: llama_base (Llama-3.1-8B), qwen3_moe (Qwen3-30B-A3B; expert inputs through
patch_qwen3_moe_experts), deepseek_moe_16b (pure-PyTorch port in deepseek_moe_model.py).
"""
import csv
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_tokenizer_and_encodings, measure_ppl  # noqa: E402,F401
from formats import quantize_tensor_config, quantize_tensor_group_topk_boost  # noqa: E402
from quant_targets import (replace_linear_modules, patch_eager, patch_eager_decomposed,  # noqa: E402
                           quantize_p_causal_full, _chunked_query, MODEL_CONFIG)

torch.manual_seed(1111)

FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, base_config="E2M1_amax_E4M3", boost="E2M3_amax_E4M3"),
    "mxfp4": dict(block_size=32, base_config="E2M1_mxfp4_ue8m0", boost="E2M3_mxfp4_ue8m0"),
}
GROUP = 16  # selection group size for magnitude / crescent
ARMS = ["bf16", "native", "magnitude", "all_e2m3", "crescent"]
LABELS = {"bf16": "BF16", "native": "Native", "magnitude": "Magnitude", "all_e2m3": "All_E2M3", "crescent": "Crescent"}
PTQ_CACHE_ROOT = os.environ.get("PTQ_CACHE_ROOT", os.path.expanduser("~/ptq_cache"))
FIELDS = ["format", "model", "setting", "ppl", "increase", "elapsed_sec"]


def cache_path_for(fmt, model_key):
    return os.path.join(PTQ_CACHE_ROOT, fmt, model_key, "ptq_cache_NVFP4.pt")  # filename is NVFP4 for both formats


def load_gptq_cache(fmt, model_key):
    path = cache_path_for(fmt, model_key)
    print(f"loading GPTQ weight cache {path}", flush=True)
    return torch.load(path, map_location="cpu")


def load_cache_into(model, cache) -> int:
    """Copy the GPTQ weights in: nn.Linear weights by module name, batched expert parameters
    (Qwen gate_up_proj/down_proj) by parameter name. Must run BEFORE Linear modules are wrapped."""
    modules = dict(model.named_modules())
    params = dict(model.named_parameters())
    n = 0
    for key, w in cache.items():
        if key in modules and hasattr(modules[key], "weight"):
            t = modules[key].weight
        elif key in params:
            t = params[key]
        else:
            raise KeyError(f"cache key {key!r} matches neither a module weight nor a parameter")
        t.data.copy_(w.to(t.device, t.dtype))
        n += 1
    assert n == len(cache)
    return n


def plain_fns(fmt, config=None):
    """(fn, p_fn): block quantization with `config` (default: the format's E2M1 base) for activations and P."""
    c = FORMAT_CONFIG[fmt]
    bs, cfg = c["block_size"], config or c["base_config"]

    def fn(x):
        return quantize_tensor_config(x.detach().to(torch.float32), cfg, bs).to(x.dtype)

    def p_fn(aw):  # head-chunked: quantize_p_causal_full materializes [heads, q_len, k_len] per chunk
        if aw.shape[1] <= 4:
            return quantize_p_causal_full(aw, cfg, bs)
        return torch.cat([quantize_p_causal_full(c_, cfg, bs) for c_ in aw.split(4, dim=1)], dim=1)
    return fn, p_fn


def selective_p(fn, block_size):
    """P with fn applied on causal positions only, head-chunked (4 heads at a time)."""
    def p_one(aw):
        k_len = aw.shape[3]

        def chunk_fn(chunk, row_offset):
            full = fn(chunk)
            rows = (torch.arange(chunk.shape[2], device=chunk.device) + row_offset).view(1, 1, -1, 1)
            cols = torch.arange(k_len, device=chunk.device).view(1, 1, 1, k_len)
            return torch.where(cols <= rows, full, chunk)
        return _chunked_query(aw, block_size, chunk_fn)

    def p_fn(aw):
        if aw.shape[1] <= 4:
            return p_one(aw)
        return torch.cat([p_one(c_) for c_ in aw.split(4, dim=1)], dim=1)
    return p_fn


def arm_setup(fmt, arm, model_key=None):
    """-> (linear_fn, attn) with attn = ('plain', q_fn, k_fn, v_fn, p_fn) or ('decomposed', base_fns, boost_fns)."""
    c = FORMAT_CONFIG[fmt]
    bs, base, boost = c["block_size"], c["base_config"], c["boost"]
    plain_fn, plain_p = plain_fns(fmt)
    if arm == "native":
        return plain_fn, ("plain", plain_fn, plain_fn, plain_fn, plain_p)
    if arm == "all_e2m3":
        fn6, p6 = plain_fns(fmt, boost)
        return fn6, ("plain", fn6, fn6, fn6, p6)
    if arm in ("crescent", "magnitude"):
        by = "error" if arm == "crescent" else "magnitude"

        def topk_fn(k):
            return lambda x: quantize_tensor_group_topk_boost(x.detach().to(torch.float32), GROUP, k, block_size=bs,
                                                              base_config=base, boost_config=boost, by=by).to(x.dtype)
        a1 = topk_fn(1)
        base_fns = {"q": plain_fn, "k": plain_fn, "v": plain_fn, "p": plain_p}
        boost_fns = {"q": a1, "k": a1, "v": a1, "p": selective_p(a1, bs)}
        return topk_fn(2), ("decomposed", base_fns, boost_fns)
    raise ValueError(f"unknown arm {arm}")


def build_model(model_key, arm, fmt, cache):
    """Returns (model, restore). Class patches are applied BEFORE loading (accelerate captures
    forwards at load time -- a patch made afterwards silently does nothing)."""
    quantized = arm != "bf16"
    if quantized:
        linear_fn, attn = arm_setup(fmt, arm, model_key)

    if model_key == "llama_base":
        from transformers import AutoModelForCausalLM
        import transformers.models.llama.modeling_llama as attn_mod
        cfg = MODEL_CONFIG[model_key]
        restore = lambda: None
        if quantized:
            if attn[0] == "plain":
                attn_cls, orig = patch_eager(cfg["arch"], attn_mod, attn[1], attn[2], attn[3], attn[4])
            else:
                attn_cls, orig = patch_eager_decomposed(cfg["arch"], attn_mod, attn[1], attn[2])
            restore = lambda: setattr(attn_cls, "forward", orig)
        kw = dict(attn_implementation="eager") if quantized else {}
        model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=torch.bfloat16,
                                                       low_cpu_mem_usage=True, **kw)
        model.to(torch.device(os.environ.get("W4A4_LLAMA_DEVICE", "cuda:0")))
    elif model_key == "qwen3_moe":
        from transformers import AutoModelForCausalLM
        from qwen3_moe_targets import MODEL_NAME, patch_eager_qwen3_moe, patch_eager_qwen3_moe_decomposed, patch_qwen3_moe_experts
        import transformers.models.qwen3_moe.modeling_qwen3_moe as attn_mod
        restore = lambda: None
        kw = {}
        if quantized:
            if attn[0] == "plain":
                attn_cls, orig_a = patch_eager_qwen3_moe(attn_mod, attn[1], attn[2], attn[3], attn[4])
            else:
                attn_cls, orig_a = patch_eager_qwen3_moe_decomposed(attn_mod, attn[1], attn[2])
            exp_cls, orig_e = patch_qwen3_moe_experts(attn_mod, linear_fn, linear_fn)
            restore = lambda: (setattr(attn_cls, "forward", orig_a), setattr(exp_cls, "forward", orig_e))
            kw = dict(attn_implementation="eager")
        # Cap each visible GPU below its full capacity: newer transformers' weight-loading
        # (core_model_loading.py's MoE expert MergeModulelist) needs extra headroom to stack
        # per-expert tensors into the batched parameter -- device_map="auto" with no cap packs
        # each GPU right up to capacity first, leaving nothing for that merge buffer (real OOM
        # during from_pretrained's weight conversion, not inference). Overridable via env.
        qwen_max_mem = os.environ.get("QWEN_EVAL_MAX_MEM_PER_GPU")
        if qwen_max_mem and torch.cuda.device_count() > 0:
            kw["max_memory"] = {i: qwen_max_mem for i in range(torch.cuda.device_count())}
            kw["max_memory"]["cpu"] = os.environ.get("QWEN_EVAL_MAX_MEM_CPU", "150GiB")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                       device_map="auto", **kw)
    elif model_key == "deepseek_moe_16b":
        import deepseek_moe_model as dsm
        restore = lambda: (dsm.set_hooks(), dsm.set_decomposed())
        if quantized:
            if attn[0] == "plain":
                dsm.set_hooks(q=attn[1], k=attn[2], v=attn[3], p=attn[4])
            else:
                dsm.set_decomposed(attn[1], attn[2])
        model = dsm.load_model()
    else:
        raise SystemExit(f"unknown model {model_key}")

    model.eval()
    if quantized:
        n_w = load_cache_into(model, cache)
        n_lin = replace_linear_modules(model, linear_fn)
        print(f"[{model_key}][{fmt}][{arm}] loaded {n_w} GPTQ weight tensors, wrapped {n_lin} Linear layers", flush=True)
    return model, restore


def model_name_of(model_key):
    if model_key == "qwen3_moe":
        from qwen3_moe_targets import MODEL_NAME
        return MODEL_NAME
    if model_key == "deepseek_moe_16b":
        from deepseek_moe_model import MODEL_NAME
        return MODEL_NAME
    return MODEL_CONFIG[model_key]["model_name"]


def run_all(fmt, model_key, arms, encodings, csv_path, cache_for=None, labels=None, **ppl_kw):
    """PPL of each arm in `arms`, appended to csv_path (rows already there are skipped).
    cache_for(): GPTQ weight dict for the quantized arms (default: this model's GPTQ cache; return {} for BF16 weights).
    labels: arm -> setting string (default LABELS). ppl_kw: passed to measure_ppl (e.g. C4's seq_len/stride/max_steps)."""
    labels = labels or LABELS
    rows = []
    if os.path.exists(csv_path):  # resume: keep finished arms
        with open(csv_path) as f:
            rows = [dict(r, ppl=float(r["ppl"]), increase=float(r["increase"])) for r in csv.DictReader(f)]
    done = {r["setting"]: r["ppl"] for r in rows}
    bf16 = done.get(labels["bf16"])
    cache = None
    for arm in arms:
        label = labels[arm]
        if label in done:
            print(f"[{fmt}][{model_key}] {label} already measured ({done[label]:.6f}) -- skipping", flush=True)
            continue
        if arm != "bf16" and cache is None:
            cache = cache_for() if cache_for else load_gptq_cache(fmt, model_key)
        same_seeds()
        model, restore = build_model(model_key, arm, fmt, cache)
        t0 = time.time()
        try:
            ppl = measure_ppl(model, encodings, f"{model_key}__{fmt}__{label}", model.device, **ppl_kw)
        finally:
            restore()
        elapsed = time.time() - t0
        del model
        cleanup_cuda()
        if arm == "bf16":
            bf16 = ppl
        print(f"[{fmt}][{model_key}] {label} PPL = {ppl:.6f}" + ("" if arm == "bf16" else f"  ({ppl - bf16:+.4f} vs BF16)")
              + f"  elapsed={elapsed:.1f}s", flush=True)
        rows.append({"format": fmt, "model": model_key, "setting": label, "ppl": ppl, "increase": ppl - bf16,
                     "elapsed_sec": round(elapsed, 1)})
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="") as f:  # after every arm: a late crash keeps earlier arms
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"Saved -> {csv_path}", flush=True)
    return rows
