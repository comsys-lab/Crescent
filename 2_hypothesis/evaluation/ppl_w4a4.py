"""W4A4 PPL: BF16 vs W4A4 baseline vs W4A4 + Ours (the single final solution, everything through H6).

  BF16          : unquantized model.
  W4A4_baseline : weights = GPTQ ptq_cache (4-bit, weight_quantization/); activations = plain block
                  quantization on Linear inputs + Q + K + V + P -- every activation operand feeding a
                  matmul is quantized (real W4A4; V/P plain, no boost -- H0/P_full confirmed both cost
                  almost nothing either way, see W4A4_Ours below).
  W4A4_Ours     : same GPTQ weights + the final technique of H0-H6 on the activations:
                    * surfaces: Linear inputs + Q + K boosted (H0); V, P plain-quantized (no boost);
                    * positions chosen by activation quantization error, not block max (H1/H3);
                    * mantissa-only boost, E2M1 -> E2M3 (H2/H4/H5); scale and exponent unchanged;
                    * grouped local selection, one largest-error element per group (H6):
                        - Linear input : group size 8  (block/8 boosted: NVFP4 2, MXFP4 4), one operand;
                        - Q and K      : group size 16 (block/16 each: NVFP4 1+1, MXFP4 2+2 in a QK^T dot
                                         product). Tightened from group 4/8 (2026-10-29): at the more
                                         generous 4/8 budget, magnitude- and error-based selection agree
                                         often enough within a small local group that W4A4_Ours and
                                         W4A4_Magnitude converge too closely to support the "error-based
                                         selection beats magnitude-based" claim; QK's group 16 matches
                                         NVFP4's own block size (16), reproducing H1's original whole-
                                         block single-pick scope where the two criteria diverge sharply.
                                         Confirmed on Llama (WikiText2): Magnitude-Ours gap widens from
                                         0.19/0.20 (NVFP4/MXFP4) to 0.31/0.24 at group 8/16.
                    * QK^T uses the error-compensation decomposition (drops the delta x delta term).
                    * V, P: plain block quantization (no boost -- H0/H0_activation_isolation's
                      P_full-corrected isolation confirmed both cost almost nothing).

  W4A4_Magnitude : prior-work comparison point -- identical to W4A4_Ours (same surfaces, same group
                  sizes, same boost precision, same QK^T decomposition, same V/P plain quantization)
                  except the ONE boosted position per group is chosen by raw |activation| magnitude
                  (prior-work-style outlier selection) instead of quantization error (H1/H3's finding).

  W4A4_Rot      : baseline + MICRO ROTATION: block-diagonal Hadamard (group = quantization block, 16 NVFP4 /
                  32 MXFP4) on every quantized surface -- Linear weights (offline, GPTQ redone in the rotated
                  space: weight_quantization/microrot_cache.py) and inputs (online), and Q/K per head after RoPE.
  W4A4_Rot_Ours : the same rotation, then Ours (grouped boost on the rotated activations).
                  R is orthonormal and applied to both operands of every matmul, so it is exact without quantization.
  W4A4_FullRot  : baseline + FULL (QuaRot-style) rotation: RMSNorm weights folded into the Linears that read them
                  (common/norm_folding.py), then a Hadamard over the WHOLE input dim (common/full_rotation.py; Paley
                  factors for widths like 14336 / 768 / 1408) on every Linear input and its weights (GPTQ redone in that
                  space: weight_quantization/fullrot_cache.py), and over head_dim on Q and K after RoPE.
  W4A4_FullRot_Ours : the same full rotation, then Ours on the rotated activations.

Models: llama_base (Llama-3.1-8B), qwen3_moe (Qwen3-30B-A3B; expert inputs are covered through
patch_qwen3_moe_experts), deepseek_moe_16b (pure-PyTorch port; experts are nn.Linear).

Resumable: arms already present in results/ppl_{fmt}_{model}.csv are skipped.
W4A4_ARMS=fullrot,fullrot_ours (env) restricts the run to those arms.

Usage: CUDA_VISIBLE_DEVICES=<GPUs> python3 ppl_w4a4.py <nvfp4|mxfp4> <llama_base|qwen3_moe|deepseek_moe_16b>
"""
import csv
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_tokenizer_and_encodings, measure_ppl
from formats import quantize_tensor_config, quantize_tensor_grouped_boost, quantize_tensor_grouped_boost_by_magnitude
from quant_targets import replace_linear_modules, patch_eager, patch_eager_decomposed, quantize_p_causal_full, MODEL_CONFIG
from micro_rotation import with_rotation
from full_rotation import with_full_rotation
from norm_folding import fold_norms

torch.manual_seed(1111)

FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, base_config="E2M1_amax_E4M3", boost="E2M3_amax_E4M3"),
    "mxfp4": dict(block_size=32, base_config="E2M1_mxfp4_ue8m0", boost="E2M3_mxfp4_ue8m0"),
}
LINEAR_DIV = 8   # Linear: group size 8 -> block/8 elements boosted (tightened from 4, see module docstring)
QK_DIV = 16      # Q and K: group size 16 -> block/16 elements boosted each (tightened from 8)
ALL_ARMS = ["bf16", "baseline", "grouped", "magnitude", "mxfp6", "rot", "rot_ours", "fullrot", "fullrot_ours"]
LABELS = {"bf16": "BF16", "baseline": "W4A4_baseline", "grouped": "W4A4_Ours", "magnitude": "W4A4_Magnitude",
          "mxfp6": "W4A4_FullBoost", "czardas_top2": "W4A4_Czardas_top2", "magnitude_top2": "W4A4_Magnitude_top2",
          "rot": "W4A4_Rot", "rot_ours": "W4A4_Rot_Ours", "fullrot": "W4A4_FullRot", "fullrot_ours": "W4A4_FullRot_Ours"}
PTQ_CACHE_ROOT = os.environ.get("PTQ_CACHE_ROOT", os.path.expanduser("~/ptq_cache"))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
PPL_KW = {}  # tests override (e.g. max_steps=1)


def cache_path_for(fmt, model_key):
    return os.path.join(PTQ_CACHE_ROOT, fmt, model_key, "ptq_cache_NVFP4.pt")  # filename is NVFP4 for both formats


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


def cache_kind(arm):
    if arm.startswith("fullrot"):
        return "fullrot"
    return "rot" if arm.startswith("rot") else "plain"


def arm_setup(fmt, arm, model_key=None):
    """-> (linear_fn, attn) where attn = ('plain', q_fn, k_fn, v_fn, p_fn) or ('decomposed', base_fns, boost_fns).
    V and P are plain-quantized (no boost -- H0/H0_activation_isolation's P_full-corrected isolation
    confirmed both cost almost nothing even with the diagonal partial block properly included, so
    there is no accuracy reason to spend a boost budget on them) in EVERY quantized arm -- "baseline"
    (Native FP4) and "grouped" (Ours) alike, so both are now real W4A4 (every activation operand
    feeding a matmul is quantized), matching how actual FP4 attention hardware/kernels work.
    Rotation arms wrap every quantizer as x -> quantizer(x R) (R = micro Hadamard, group = block size)
    -- V/P are left unrotated (unchanged from before), since P isn't a rotatable vector-space quantity
    the same way Q/K/Linear-input are.
    """
    if arm.startswith("fullrot"):
        linear_fn, attn = arm_setup(fmt, "baseline" if arm == "fullrot" else "grouped")
        # Qwen3-MoE only: the post-RoPE head_dim Hadamard rotation on Q/K, combined with
        # Qwen3MoeAttention's QK-Norm (RMSNorm on Q/K before RoPE -- no analog in Llama/DeepSeek),
        # produces just enough extra quantization noise (confirmed empirically: ~10-17% relative
        # error on the QK^T GEMM output in isolation, vs ~4-5% for the Linear-weight-rotation path
        # alone) to regularly flip Qwen's discrete top-k MoE routing decisions. Qwen3-30B-A3B has
        # NO dense buffer layers (every one of its 48 layers routes), so this compounds layer over
        # layer -- measured 28-30% of tokens routed to a different expert set than BF16 already at
        # layer 0, saturating to 100% by layer ~26-32, which is exactly what produced the garbage
        # zero-shot/PPL (near-chance accuracy, PPL in the thousands) for Qwen Rotation specifically.
        # Boosting Q/K precision (fullrot_ours) barely moved this (28.7% at layer 0 vs 30.3%, same
        # saturation pattern) -- ruling out "needs more mantissa bits" and pointing at the rotation
        # step itself. Fix: leave Q/K at the base arm's plain (unrotated) quantization for Qwen,
        # keep Linear GEMM (weights + inputs) fully rotated as normal -- the Linear-only path tested
        # clean (4-5% error, matching the healthy Llama/DeepSeek Rotation arms). Llama/DeepSeek are
        # unaffected (no QK-Norm, no all-MoE depth) and keep full QK rotation.
        qk_rot = (lambda f: f) if model_key == "qwen3_moe" else with_full_rotation
        if attn[0] == "plain":
            return with_full_rotation(linear_fn), ("plain", qk_rot(attn[1]), qk_rot(attn[2]), attn[3], attn[4])
        base_fns = dict(attn[1], q=qk_rot(attn[1]["q"]), k=qk_rot(attn[1]["k"]))
        boost_fns = dict(attn[2], q=qk_rot(attn[2]["q"]), k=qk_rot(attn[2]["k"]))
        return with_full_rotation(linear_fn), ("decomposed", base_fns, boost_fns)
    if arm.startswith("rot"):
        g = FORMAT_CONFIG[fmt]["block_size"]
        linear_fn, attn = arm_setup(fmt, "baseline" if arm == "rot" else "grouped")
        rot = lambda f: with_rotation(f, g)
        if attn[0] == "plain":
            return rot(linear_fn), ("plain", rot(attn[1]), rot(attn[2]), attn[3], attn[4])
        base_fns = dict(attn[1], q=rot(attn[1]["q"]), k=rot(attn[1]["k"]))
        boost_fns = dict(attn[2], q=rot(attn[2]["q"]), k=rot(attn[2]["k"]))
        return rot(linear_fn), ("decomposed", base_fns, boost_fns)
    c = FORMAT_CONFIG[fmt]
    bs, base = c["block_size"], c["base_config"]

    def plain_fn(x):
        return quantize_tensor_config(x.detach().to(torch.float32), base, bs).to(x.dtype)

    def plain_p(aw):
        # head-chunked: quantize_p_causal_full materializes the full [heads,q_len,k_len] tensor at
        # once per chunk, which OOM'd repeatedly on big MoE models at full head count (same issue
        # H0_activation_isolation's plain_p hit) -- chunk 4 heads at a time as a precaution.
        if aw.shape[1] <= 4:
            return quantize_p_causal_full(aw, base, bs)
        return torch.cat([quantize_p_causal_full(c_, base, bs) for c_ in aw.split(4, dim=1)], dim=1)

    if arm == "baseline":
        return plain_fn, ("plain", plain_fn, plain_fn, plain_fn, plain_p)

    boost = c["boost"]

    def grouped(group):
        return lambda x: quantize_tensor_grouped_boost(x.detach().to(torch.float32), group, block_size=bs,
                                                        base_config=base, boost_config=boost).to(x.dtype)

    if arm == "magnitude":
        def grouped_mag(group):
            return lambda x: quantize_tensor_grouped_boost_by_magnitude(x.detach().to(torch.float32), group, block_size=bs,
                                                                         base_config=base, boost_config=boost).to(x.dtype)
        linear_fn, qk_fn = grouped_mag(LINEAR_DIV), grouped_mag(QK_DIV)
        base_fns = {"q": plain_fn, "k": plain_fn, "v": plain_fn, "p": plain_p}
        boost_fns = {"q": qk_fn, "k": qk_fn, "v": plain_fn, "p": plain_p}
        return linear_fn, ("decomposed", base_fns, boost_fns)

    # MX+ (Lee et al., MICRO'25), as specified in the paper -- see common/m2xfp.py:mxplus. BM boosted on
    # every dot-product operand (Linear, Q, K, V, P); plain attention matmuls (BCU computes BM x BM exactly).
    if arm == "mxfp4plus":
        from m2xfp import mxplus
        from quant_targets import _chunked_query
        mp = lambda x: mxplus(x.detach().to(torch.float32), fmt).to(x.dtype)

        def p_one(aw):
            k_len = aw.shape[3]
            def fn(chunk, row_offset):
                full = mp(chunk)
                rows = (torch.arange(chunk.shape[2], device=chunk.device) + row_offset).view(1, 1, -1, 1)
                cols = torch.arange(k_len, device=chunk.device).view(1, 1, 1, k_len)
                return torch.where(cols <= rows, full, chunk)
            return _chunked_query(aw, bs, fn)

        def p_fn(aw):
            if aw.shape[1] <= 4:
                return p_one(aw)
            return torch.cat([p_one(c_) for c_ in aw.split(4, dim=1)], dim=1)
        return mp, ("plain", mp, mp, mp, p_fn)

    # M2XFP (Hu et al., ASPLOS'26), as specified in the paper -- see common/m2xfp.py:
    #   Linear inputs, Q, P : Elem-EM-top1 (Alg. 1; FP4-domain top-1, +1 bias / clamp 2-bit encoding)
    #   K, V                : Sg-EM-2bit adaptive (Sec 6.4 "Extension to Attention and KV Cache")
    #   MXFP4 group/subgroup 32/8 with OCP floor E8M0 scale; NVFP4 = the paper's M2-NVFP4 (16/4, E4M3 scale).
    # Plain (non-decomposed) attention: M2XFP's PE consumes the decoded values directly.
    if arm == "m2xfp":
        from m2xfp import m2xfp_elem_em, m2xfp_sg_em
        from quant_targets import _chunked_query
        elem = lambda x: m2xfp_elem_em(x.detach().to(torch.float32), fmt).to(x.dtype)
        sg = lambda x: m2xfp_sg_em(x.detach().to(torch.float32), fmt).to(x.dtype)

        def p_one(aw):
            k_len = aw.shape[3]
            def fn(chunk, row_offset):
                full = elem(chunk)
                rows = (torch.arange(chunk.shape[2], device=chunk.device) + row_offset).view(1, 1, -1, 1)
                cols = torch.arange(k_len, device=chunk.device).view(1, 1, 1, k_len)
                return torch.where(cols <= rows, full, chunk)
            return _chunked_query(aw, bs, fn)

        def p_fn(aw):
            if aw.shape[1] <= 4:
                return p_one(aw)
            return torch.cat([p_one(c_) for c_ in aw.split(4, dim=1)], dim=1)
        return elem, ("plain", elem, sg, sg, p_fn)

    # Final Czardas / Magnitude config (2026-11-07): Linear = exact top-2 per 16-element group; Q/K/V/P = top-1
    # per 16 (decomposed QK^T and P@V, delta x delta dropped). Selection by squared error (czardas_top2) or
    # |a| (magnitude_top2). Identical to evaluation/ppl_czardas_top2.py. V along head_dim (V-axis deferred).
    if arm in ("czardas_top2", "magnitude_top2"):
        from formats import quantize_tensor_group_topk_boost
        from quant_targets import _chunked_query
        by = "error" if arm == "czardas_top2" else "magnitude"

        def topk_fn(k):
            return lambda x: quantize_tensor_group_topk_boost(x.detach().to(torch.float32), 16, k, block_size=bs,
                                                              base_config=base, boost_config=boost, by=by).to(x.dtype)
        a1 = topk_fn(1)

        def p_boost_one(aw):
            k_len = aw.shape[3]
            def fn(chunk, row_offset):
                full = a1(chunk)
                rows = (torch.arange(chunk.shape[2], device=chunk.device) + row_offset).view(1, 1, -1, 1)
                cols = torch.arange(k_len, device=chunk.device).view(1, 1, 1, k_len)
                return torch.where(cols <= rows, full, chunk)
            return _chunked_query(aw, bs, fn)

        def p_boost(aw):
            if aw.shape[1] <= 4:
                return p_boost_one(aw)
            return torch.cat([p_boost_one(c_) for c_ in aw.split(4, dim=1)], dim=1)
        base_fns = {"q": plain_fn, "k": plain_fn, "v": plain_fn, "p": plain_p}
        boost_fns = {"q": a1, "k": a1, "v": a1, "p": p_boost}
        return topk_fn(2), ("decomposed", base_fns, boost_fns)

    if arm == "mxfp6":
        def plain6_fn(x):
            return quantize_tensor_config(x.detach().to(torch.float32), boost, bs).to(x.dtype)
        def plain6_p(aw):
            if aw.shape[1] <= 4:
                return quantize_p_causal_full(aw, boost, bs)
            return torch.cat([quantize_p_causal_full(c_, boost, bs) for c_ in aw.split(4, dim=1)], dim=1)
        return plain6_fn, ("plain", plain6_fn, plain6_fn, plain6_fn, plain6_p)

    linear_fn, qk_fn = grouped(LINEAR_DIV), grouped(QK_DIV)  # group SIZE 8 (Linear) / 16 (Q, K)
    base_fns = {"q": plain_fn, "k": plain_fn, "v": plain_fn, "p": plain_p}
    boost_fns = {"q": qk_fn, "k": qk_fn, "v": plain_fn, "p": plain_p}  # v/p: base==boost -> decomposition
    return linear_fn, ("decomposed", base_fns, boost_fns)              # reduces to plain p_q @ v_q, no cross term


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
        if arm.startswith("fullrot"):
            fold_norms(model)  # norms -> 1, gamma folded into the readers (router, lm_head included) BEFORE the cache is loaded
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


def run_all(fmt, model_key, cache_for, encodings, csv_path):
    """cache_for(kind) -> GPTQ weight dict for kind 'plain' | 'rot'; called lazily, only for arms still to run."""
    rows = []
    if os.path.exists(csv_path):  # resume: keep finished arms
        with open(csv_path) as f:
            rows = [dict(r, ppl=float(r["ppl"]), increase=float(r["increase"])) for r in csv.DictReader(f)]
    done = {r["setting"]: r["ppl"] for r in rows}
    bf16 = done.get(LABELS["bf16"])
    only = [a for a in os.environ.get("W4A4_ARMS", "").split(",") if a]  # optional: run just these arms
    for arm in ALL_ARMS:
        if only and arm not in only:
            continue
        if LABELS[arm] in done:
            print(f"[{fmt}][{model_key}] {LABELS[arm]} already measured ({done[LABELS[arm]]:.6f}) -- skipping", flush=True)
            continue
        same_seeds()
        cache = cache_for(cache_kind(arm)) if arm != "bf16" else None
        model, restore = build_model(model_key, arm, fmt, cache)
        label = LABELS[arm]
        t0 = time.time()
        try:
            ppl = measure_ppl(model, encodings, f"{model_key}__{fmt}__{label}", model.device, **PPL_KW)
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
        with open(csv_path, "w", newline="") as f:  # after every arm: a late crash keeps earlier arms
            w = csv.DictWriter(f, fieldnames=["format", "model", "setting", "ppl", "increase", "elapsed_sec"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Saved -> {csv_path}", flush=True)
    return rows


def make_cache_provider(fmt, model_key):
    loaded = {}

    def cache_for(kind):
        if kind not in loaded:
            loaded.clear()  # hold one cache at a time (Qwen's is 60 GB of host RAM)
            if kind == "plain":
                path = cache_path_for(fmt, model_key)
                print(f"loading GPTQ weight cache {path}", flush=True)
                loaded[kind] = torch.load(path, map_location="cpu")
            elif kind == "fullrot":  # fully rotated (QuaRot-style) GPTQ cache
                sys.path.insert(0, os.path.join(HERE, "..", "weight_quantization"))
                from fullrot_cache import load_or_build as load_or_build_full
                loaded[kind] = load_or_build_full(fmt, model_key)
            else:  # micro-rotated GPTQ cache: loaded, or built with the visible GPUs (and saved if the disk has room)
                sys.path.insert(0, os.path.join(HERE, "..", "weight_quantization"))
                from microrot_cache import load_or_build
                loaded[kind] = load_or_build(fmt, model_key)
        return loaded[kind]

    return cache_for


def main():
    fmt, model_key = sys.argv[1], sys.argv[2]
    same_seeds()
    _, encodings = load_tokenizer_and_encodings(model_name_of(model_key), load_wikitext2_text())
    run_all(fmt, model_key, make_cache_provider(fmt, model_key), encodings,
            os.path.join(RESULTS_DIR, f"ppl_{fmt}_{model_key}.csv"))


if __name__ == "__main__":
    main()
