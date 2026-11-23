"""Weight GPTQ + Hessian-weighted scaling-factor search -- WEIGHT
quantization only. No rotation, no activation boost, no QK handling
(rotation will be added later; see weight_gptq_llama.py, the rotation+
boost 8-arm Llama-only reference this was distilled from).

Two-level scale search per weight tensor, both levels chosen to minimize
Hessian-diagonal-weighted squared error (same recipe as weight_gptq_llama.
py's block_gptq_quantize_weight(use_hessiandiag_sT=True) / find_best_s_T_
hessiandiag / quantize_*_blocks_hessiandiag, ported here on top of this
project's own FP4 primitives instead of duplicating them):
  1. s_T (NVFP4 only): ONE global scale multiplier for the whole weight
     tensor, found by scanning log2 octaves around a reference scale.
  2. per-block scale: block_size (16 NVFP4 / 32 MXFP4) columns at a time,
     scanning e4m3-rounded (NVFP4) or ue8m0 octave-shifted (MXFP4)
     candidates on top of s_T.
Between blocks, the standard GPTQ/OBS step propagates each block's
quantization error into the not-yet-quantized columns via the Hessian
inverse (Cholesky-based), before those columns are calibrated.

Calibration/eval split: unlike weight_gptq_llama.py (which calibrates and
evaluates PPL on the SAME wikitext2 test-split text), this uses eval_utils.
load_wikitext2_calib_text() (train split) for Hessian collection and
load_wikitext2_text() (test split) for PPL -- matching this project's
established train/test separation (see eval_utils.py's own comment on
why) instead of the older script's overlap.

Output: ptq_cache/{fmt}/{model_key}/ptq_cache_NVFP4.pt -- path AND
filename convention (filename literally always says "NVFP4" regardless of
fmt) match common/quant_targets.py's CACHE_PATH_TEMPLATE/
load_weights_from_cache exactly, so H0_activation_isolation/weight_only.py
can load these caches with zero changes.

Qwen3-30B-A3B is out of scope here: its MoE experts (gate_up_proj/
down_proj) are batched nn.Parameter tensors, not nn.Linear -- GPTQ's
column-block OBS update assumes a 2D [out_features, in_features] nn.Linear
weight, which doesn't apply the same way to a [num_experts, ...] batched
tensor (per-expert calibration would need separate Hessians gathered only
from that expert's routed tokens). Left for later, same as the activation
scripts (see common/qwen3_moe_targets.py's module docstring).

Usage: python3 weight_gptq.py <nvfp4|mxfp4> [model ...]   (default: llama ministral)
"""
import csv
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_wikitext2_calib_text, load_tokenizer_and_encodings, measure_ppl
from quant import get_positive_fp_levels, quantize_to_fp_levels, block_last_dimension
from formats import FMT_E2M1, round_scale
from quant_targets import MODEL_CONFIG, should_skip_linear

torch.manual_seed(1111)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GPTQ_CPU = os.environ.get("GPTQ_CPU", "1") != "0"  # Hessian accumulation on CPU (saves GPU memory, cheap relative to the model forward passes)

FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, scale_format="e4m3", has_tensor_scale=True),
    "mxfp4": dict(block_size=32, scale_format="ue8m0", has_tensor_scale=False),
}
E4M3_MULTS = [2.0 ** (k / 8.0) for k in range(-32, 17)]  # per-block candidate multipliers on top of s_T (NVFP4)
UE8M0_OCTAVE_SHIFTS = [-3, -2, -1, 0, 1]  # per-block candidate shifts on the round-up base scale (MXFP4)
S_T_LOG2_OCTAVES = range(-24, 25)
S_T_SAMPLE_BLOCKS = 100_000

GPTQ_CALIB_SEQ_LEN = int(os.environ.get("GPTQ_CALIB_SEQ_LEN", "512"))
GPTQ_CALIB_STRIDE = int(os.environ.get("GPTQ_CALIB_STRIDE", "512"))
GPTQ_CALIB_STEPS = int(os.environ.get("GPTQ_CALIB_STEPS", "8"))
GPTQ_DAMP_PERCENT = float(os.environ.get("GPTQ_DAMP_PERCENT", "0.01"))
SKIP_PPL = os.environ.get("GPTQ_SKIP_PPL", "0") == "1"  # 1: only build the ptq_cache, no PPL measurement

PTQ_CACHE_ROOT = os.environ.get("PTQ_CACHE_ROOT", os.path.expanduser("~/ptq_cache"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def cache_path_for(fmt: str, model_key: str) -> str:
    # filename always literally "ptq_cache_NVFP4.pt" regardless of fmt -- matches
    # common/quant_targets.py's CACHE_PATH_TEMPLATE quirk exactly, see module docstring.
    return os.path.join(PTQ_CACHE_ROOT, fmt, model_key, "ptq_cache_NVFP4.pt")


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    m = model
    for part in name.split("."):
        m = getattr(m, part)
    return m


def iter_target_linear_names(model: nn.Module):
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and not should_skip_linear(n)]


# --------------------------------------------------------- FP4 quantize ---

def quantize_blocks_hessiandiag(blocks: torch.Tensor, h_diag_block: torch.Tensor, fmt_cfg: dict, s_T: float) -> torch.Tensor:
    """Per-block scale search (NVFP4: 49 e4m3 candidates on top of s_T;
    MXFP4: 5 ue8m0 octave-shift candidates, s_T unused), objective =
    Hessian-diagonal-weighted sum of squared error."""
    levels, lsb_even = get_positive_fp_levels(FMT_E2M1, blocks.device)
    fp4_max = levels[-1].to(device=blocks.device, dtype=blocks.dtype)
    max_abs = blocks.abs().amax(dim=1, keepdim=True)
    h_row = h_diag_block.to(device=blocks.device, dtype=blocks.dtype).view(1, -1)

    if fmt_cfg["has_tensor_scale"]:
        ideal_scale = (max_abs / fp4_max) / s_T
        candidates = [(ideal_scale * mult, s_T) for mult in E4M3_MULTS]
    else:
        base_scale = torch.where(max_abs > 0, round_scale(max_abs / fp4_max, "ue8m0"), torch.ones_like(max_abs))
        candidates = [(base_scale * (2.0 ** shift), 1.0) for shift in UE8M0_OCTAVE_SHIFTS]

    best_err, best_q = None, None
    for cand_ideal, s_t_mult in candidates:
        if fmt_cfg["has_tensor_scale"]:
            cand_scale = torch.where(max_abs > 0, torch.clamp(round_scale(cand_ideal, "e4m3") * s_t_mult, min=2.0 ** -24), torch.ones_like(max_abs))
        else:
            cand_scale = torch.clamp(cand_ideal, min=2.0 ** -127)
        q = quantize_to_fp_levels(blocks / cand_scale, levels, lsb_even) * cand_scale
        err = ((blocks - q).pow(2) * h_row).sum(dim=1, keepdim=True)
        if best_err is None:
            best_err, best_q = err, q
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_q = torch.where(better, q, best_q)
    return best_q


def find_best_s_T(weight: torch.Tensor, hessian: torch.Tensor, block_size: int) -> float:
    """NVFP4-only: search the ONE tensor-level s_T minimizing total
    Hessian-weighted error on a sample of this weight's blocks."""
    device = weight.device
    in_f = weight.shape[1]
    n_blocks_per_row = in_f // block_size
    h_diag = torch.diag(hessian).to(device, torch.float32)
    h_diag_blocks = h_diag[:n_blocks_per_row * block_size].reshape(n_blocks_per_row, block_size)

    w_blocks_all = weight[:, :n_blocks_per_row * block_size].reshape(-1, n_blocks_per_row, block_size).to(device, torch.float32)
    out_features = w_blocks_all.shape[0]
    h_tiled = h_diag_blocks.unsqueeze(0).expand(out_features, -1, -1)

    blocks_all, h_all = w_blocks_all.reshape(-1, block_size), h_tiled.reshape(-1, block_size)
    n_total = blocks_all.shape[0]
    if n_total > S_T_SAMPLE_BLOCKS:
        idx = torch.randperm(n_total, device=device)[:S_T_SAMPLE_BLOCKS]
        blocks, h_sample = blocks_all[idx], h_all[idx]
    else:
        blocks, h_sample = blocks_all, h_all

    levels, lsb_even = get_positive_fp_levels(FMT_E2M1, blocks.device)
    fp4_max = levels[-1].to(device=blocks.device, dtype=blocks.dtype)
    max_abs = blocks.abs().amax(dim=1, keepdim=True)
    global_amax = torch.clamp(blocks.abs().amax(), min=2.0 ** -24)
    s_T_ref = (global_amax / (448.0 * fp4_max)).item()  # 448.0 = FP8 E4M3's own max finite magnitude
    s_T_candidates = [s_T_ref * (2.0 ** k) for k in S_T_LOG2_OCTAVES] + [1.0]

    best_total, best_s_T = None, 1.0
    for s_T in s_T_candidates:
        ideal_scale = (max_abs / fp4_max) / s_T
        best_err_this = None
        for mult in E4M3_MULTS:
            cand_scale = torch.where(max_abs > 0, torch.clamp(round_scale(ideal_scale * mult, "e4m3") * s_T, min=2.0 ** -24), torch.ones_like(max_abs))
            q = quantize_to_fp_levels(blocks / cand_scale, levels, lsb_even) * cand_scale
            err_sum = ((blocks - q).pow(2) * h_sample).sum(dim=1, keepdim=True)
            best_err_this = err_sum if best_err_this is None else torch.minimum(best_err_this, err_sum)
        total = best_err_this.sum().item()
        if best_total is None or total < best_total - 1e-9:
            best_total, best_s_T = total, s_T
    return best_s_T


# ------------------------------------------------------------- GPTQ/OBS ---

@torch.no_grad()
def collect_hessian(model: nn.Module, module: nn.Linear, calib_ids: torch.Tensor) -> torch.Tensor:
    h_cpu, count = None, 0

    def hook(_mod, inputs):
        nonlocal h_cpu, count
        x = inputs[0].detach()
        if not torch.is_floating_point(x):
            return
        x2 = x.reshape(-1, x.shape[-1]).to(torch.float32)
        h = x2.cpu().t().matmul(x2.cpu()) if GPTQ_CPU else x2.t().matmul(x2).cpu()
        h_cpu = h if h_cpu is None else h_cpu.add_(h)
        count += int(x2.shape[0])
        del x2

    handle = module.register_forward_pre_hook(hook)
    try:
        for begin in range(0, calib_ids.shape[1], GPTQ_CALIB_STRIDE):
            end = min(begin + GPTQ_CALIB_SEQ_LEN, calib_ids.shape[1])
            if end <= begin:
                break
            model(input_ids=calib_ids[:, begin:end].to(DEVICE), use_cache=False)
            if (begin // GPTQ_CALIB_STRIDE) + 1 >= GPTQ_CALIB_STEPS:
                break
    finally:
        handle.remove()
    if h_cpu is None or count <= 0:
        raise RuntimeError("No calibration activations collected")
    return (h_cpu / float(count)).contiguous()


def collect_hessians_batch(model: nn.Module, named_modules: dict, calib_ids: torch.Tensor) -> dict:
    """Same accumulation as collect_hessian, but for MANY target modules at
    once, in a SINGLE calibration pass -- hooks every module in
    named_modules simultaneously instead of calling collect_hessian in a
    loop (one call per name), which re-runs the full GPTQ_CALIB_STEPS-step
    calibration forward from scratch for EVERY target layer. For a model
    with N target Linears that's N x GPTQ_CALIB_STEPS full-model forward
    passes instead of just GPTQ_CALIB_STEPS -- fine for a single-GPU dense
    8B model (weight_gptq.py's own build_cache still does it the slow way,
    already validated and not worth touching), but for a 3-GPU-sharded 30B
    MoE model the per-forward-pass cross-GPU overhead compounds badly: a
    real run of weight_gptq_qwen3_moe.py crawled through only 32/192 Q/K/V/
    O layers in over an hour of wall time before this fix, confirmed via
    nvidia-smi that the GPU was genuinely busy (not idle/CPU-bound) the
    whole time -- the redundant-forward-passes architecture was the actual
    bottleneck, not underutilized hardware. named_modules: {name: module}.
    Returns {name: averaged_hessian}."""
    h_cpu: dict = {}
    count: dict = {}

    def make_hook(name):
        def hook(_mod, inputs):
            x = inputs[0].detach()
            if not torch.is_floating_point(x):
                return
            x2 = x.reshape(-1, x.shape[-1]).to(torch.float32)
            h = x2.cpu().t().matmul(x2.cpu()) if GPTQ_CPU else x2.t().matmul(x2).cpu()
            if name not in h_cpu:
                h_cpu[name] = h
                count[name] = int(x2.shape[0])
            else:
                h_cpu[name].add_(h)
                count[name] += int(x2.shape[0])
            del x2
        return hook

    handles = [m.register_forward_pre_hook(make_hook(n)) for n, m in named_modules.items()]
    try:
        for begin in range(0, calib_ids.shape[1], GPTQ_CALIB_STRIDE):
            end = min(begin + GPTQ_CALIB_SEQ_LEN, calib_ids.shape[1])
            if end <= begin:
                break
            model(input_ids=calib_ids[:, begin:end].to(DEVICE), use_cache=False)
            if (begin // GPTQ_CALIB_STRIDE) + 1 >= GPTQ_CALIB_STEPS:
                break
    finally:
        for handle in handles:
            handle.remove()
    missing = [n for n in named_modules if n not in h_cpu or count[n] <= 0]
    if missing:
        raise RuntimeError(f"No calibration activations collected for {len(missing)} modules, e.g. {missing[:3]}")
    return {n: (h_cpu[n] / float(count[n])).contiguous() for n in named_modules}


GPTQ_COND_THRESHOLD = float(os.environ.get("GPTQ_COND_THRESHOLD", "200"))  # see gptq_quantize_weight docstring -- calibrated
# against Ministral-8B's down_proj layer 2: the one bad block there has cond(h_bb)=2237 vs every other
# block's 1.6-7.2 (a >300x gap), so 200 safely separates them with wide margin on both sides.


@torch.no_grad()
def gptq_quantize_weight(weight: torch.Tensor, hessian: torch.Tensor, fmt: str) -> torch.Tensor:
    """OBS/GPTQ column-block loop. Before propagating a block's quantization
    error into later columns (w[:, end:] -= error @ solve(h_bb, h_br)), checks
    h_bb's condition number and SKIPS propagation for that one block if it's
    too ill-conditioned to invert stably.

    Found on Ministral-8B-Instruct-2410's down_proj: that layer's calibration
    Hessian has condition number ~1.2e6 (diag ranges from 1.7e-4 to 345) --
    high enough that torch.linalg.solve(h_bb, h_br) on some 16-column
    sub-blocks returns huge, effectively-arbitrary values, injecting a
    catastrophic single-element error (comparable in magnitude to the
    weight's own absmax) into a handful of rows for later columns -- while
    barely moving the tensor's aggregate relative MSE (order 0.01-0.03),
    so it's invisible to a whole-tensor error check but blows up end-to-end
    PPL (BF16 5.98 -> 8.85 measured). Raising GPTQ_DAMP_PERCENT alone does
    NOT fix this (tested 0.01/0.05/0.1/0.3 -- average error improves, the
    worst single-element error does not go away). Llama-3.1-8B-Instruct's
    down_proj does not hit this (no ill-conditioned blocks there), so this
    is model/layer-specific, not a general problem with every down_proj.

    The block's OWN quantization is unaffected by this guard -- it still
    goes through the same Hessian-diagonal-weighted scale search as every
    other block. Only the cross-block OBS error-compensation step (the part
    that's unstable here) is skipped for that specific block."""
    fmt_cfg = FORMAT_CONFIG[fmt]
    block_size = fmt_cfg["block_size"]
    _, in_features = weight.shape
    work_device = torch.device("cpu") if GPTQ_CPU else weight.device
    w = weight.detach().to(work_device, torch.float32).clone()
    h = hessian.to(work_device, torch.float32).clone()

    diag = torch.diag(h)
    s_T = find_best_s_T(w, h, block_size) if fmt_cfg["has_tensor_scale"] else 1.0
    damp = GPTQ_DAMP_PERCENT * torch.clamp(diag.mean(), min=1e-12)
    idx = torch.arange(in_features, device=work_device)
    h[idx, idx] += damp
    hessian_inv = torch.cholesky_inverse(torch.linalg.cholesky(h))

    q = torch.empty_like(w)
    n_skipped = 0
    for start in range(0, in_features, block_size):
        end = min(start + block_size, in_features)
        w_block = w[:, start:end]
        q_block = quantize_blocks_hessiandiag(w_block, diag[start:end], fmt_cfg, s_T).to(work_device, torch.float32)
        q[:, start:end] = q_block
        if end < in_features:
            h_bb = hessian_inv[start:end, start:end]
            cond = torch.linalg.cond(h_bb.double()).item()
            if cond > GPTQ_COND_THRESHOLD:
                n_skipped += 1
            else:
                error = w_block - q_block
                h_br = hessian_inv[start:end, end:]
                w[:, end:] -= error.matmul(torch.linalg.solve(h_bb, h_br))
        del w_block, q_block
    if n_skipped:
        print(f"[gptq] skipped OBS error propagation for {n_skipped}/{in_features // block_size} "
              f"ill-conditioned blocks (cond > {GPTQ_COND_THRESHOLD:.0e})", flush=True)
    return q.to(weight.device, weight.dtype)


@torch.no_grad()
def build_cache(fmt: str, model_key: str, calib_ids: torch.Tensor) -> dict:
    model_name = MODEL_CONFIG[model_key]["model_name"]
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to(DEVICE)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    target_names = iter_target_linear_names(model)
    cache: dict = {}
    for idx, name in enumerate(target_names, 1):
        module = get_module_by_name(model, name)
        if idx % 32 == 0 or idx == len(target_names):
            print(f"[{fmt}][{model_key}] GPTQ {idx}/{len(target_names)} {name}", flush=True)
        hessian = collect_hessian(model, module, calib_ids)
        q_weight = gptq_quantize_weight(module.weight.data, hessian, fmt)
        cache[name] = q_weight.cpu()
        del hessian, q_weight
        cleanup_cuda()

    del model
    cleanup_cuda()
    return cache


@torch.no_grad()
def measure_weight_only(fmt: str, model_key: str, cache: dict, encodings) -> dict:
    model_name = MODEL_CONFIG[model_key]["model_name"]
    from transformers import AutoModelForCausalLM

    same_seeds()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to(DEVICE); model.eval()
    bf16_ppl = measure_ppl(model, encodings, f"{model_key}__BF16", DEVICE)
    del model
    cleanup_cuda()

    same_seeds()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to(DEVICE); model.eval()
    modules = dict(model.named_modules())
    n = 0
    for name, w in cache.items():
        modules[name].weight.data.copy_(w.to(DEVICE, torch.bfloat16))
        n += 1
    weight_ppl = measure_ppl(model, encodings, f"{model_key}__Weight_ONLY", DEVICE)
    del model
    cleanup_cuda()

    print(f"[{fmt}][{model_key}] BF16={bf16_ppl:.6f}  Weight_ONLY={weight_ppl:.6f} (+{weight_ppl - bf16_ppl:.4f})  n_layers={n}", flush=True)
    return {"format": fmt, "model": model_key, "bf16_ppl": bf16_ppl, "weight_only_ppl": weight_ppl,
            "increase": weight_ppl - bf16_ppl, "n_layers": n}


def main():
    fmt = sys.argv[1]
    model_keys = sys.argv[2:] if len(sys.argv) > 2 else ["llama_base"]

    same_seeds()
    calib_text = load_wikitext2_calib_text()

    rows = []
    for model_key in model_keys:
        path = cache_path_for(fmt, model_key)
        if os.path.exists(path):
            print(f"[{fmt}][{model_key}] cache already exists -> {path} (loading, not rebuilding)", flush=True)
            cache = torch.load(path, map_location="cpu")
        else:
            model_name = MODEL_CONFIG[model_key]["model_name"]
            _, calib_encodings = load_tokenizer_and_encodings(model_name, calib_text)
            t0 = time.time()
            cache = build_cache(fmt, model_key, calib_encodings.input_ids)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(cache, path)
            print(f"[{fmt}][{model_key}] cache saved -> {path} ({time.time() - t0:.1f}s)", flush=True)

        if SKIP_PPL:  # cache-only run: no PPL measurement
            del cache
            cleanup_cuda()
            continue
        full_text = load_wikitext2_text()
        _, encodings = load_tokenizer_and_encodings(MODEL_CONFIG[model_key]["model_name"], full_text)
        rows.append(measure_weight_only(fmt, model_key, cache, encodings))
        del cache
        cleanup_cuda()

    if not rows:
        return
    csv_path = os.path.join(RESULTS_DIR, f"weight_gptq_ppl_{fmt}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["format", "model", "bf16_ppl", "weight_only_ppl", "increase", "n_layers"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Saved -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
