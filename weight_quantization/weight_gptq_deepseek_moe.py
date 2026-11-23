"""Weight GPTQ + Hessian-weighted scale search for DeepSeek-MoE-16B-Base -- builds the ptq_cache only.

Same algorithm as weight_gptq.py (gptq_quantize_weight imported UNCHANGED: two-level scale search,
GPTQ/OBS block propagation, ill-conditioning guard), applied to EVERY nn.Linear of the model except
lm_head: attention q/k/v/o, the dense layer-0 MLP, the 2 shared experts and all 64 routed experts of
each of the 27 MoE layers (5380 matrices). The router (bare Parameter) is not quantized, as in the
activation-side scripts. Model = common/deepseek_moe_model.py (pure-PyTorch port of the checkpoint).

Calibration: ONE forward pass over WikiText-2 *train* text (GPTQ_CALIB_STEPS x GPTQ_CALIB_SEQ_LEN tokens,
same defaults as the dense/Qwen scripts), hooking every Linear input:
  - q/k/v share one input -> one Hessian; gate/up share one input -> one Hessian (identical to
    computing it per Linear, without repeating the work);
  - routed experts get a PER-EXPERT Hessian from only the tokens routed to them; an expert that
    received no calibration tokens falls back to an identity Hessian (plain per-block quantize, no
    cross-block instability), exactly like weight_gptq_qwen3_moe.py.
Then the model is freed and the (weight, Hessian) jobs run through gptq_parallel.run_jobs -- several
worker processes per GPU, because one matrix is latency-bound (~3-5 s NVFP4 / ~0.3 s MXFP4).

Cache: {module name: dequantized bf16 weight}, keyed by nn.Linear name, at
ptq_cache/{fmt}/deepseek_moe_16b/ptq_cache_NVFP4.pt (the filename says NVFP4 for both formats, as for the
other models). Because every quantized tensor is a plain nn.Linear weight, common/quant_targets.
load_weights_from_cache loads it unchanged.

Usage: CUDA_VISIBLE_DEVICES=<free GPUs> python3 weight_gptq_deepseek_moe.py <nvfp4|mxfp4>
Env: GPTQ_PROCS_PER_DEVICE (default 6), GPTQ_CALIB_* (see weight_gptq.py)
"""
import csv
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("GPTQ_CPU", "0")  # GPTQ math on the GPU, Hessian matmul on the GPU too
from weight_gptq import GPTQ_CALIB_SEQ_LEN, GPTQ_CALIB_STRIDE, GPTQ_CALIB_STEPS  # noqa: E402
from gptq_parallel import run_jobs  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_wikitext2_calib_text, load_tokenizer_and_encodings, measure_ppl  # noqa: E402
import deepseek_moe_model as dsm  # noqa: E402

torch.manual_seed(1111)

PTQ_CACHE_ROOT = os.environ.get("PTQ_CACHE_ROOT", os.path.expanduser("~/ptq_cache"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SHARED_INPUT_LEADER = {"k_proj": "q_proj", "v_proj": "q_proj", "up_proj": "gate_proj"}  # same input as the leader


def cache_path_for(fmt: str) -> str:
    return os.path.join(PTQ_CACHE_ROOT, fmt, "deepseek_moe_16b", "ptq_cache_NVFP4.pt")


def leader_name(name: str) -> str:
    """Name of the module whose input Hessian this Linear shares (itself if it has its own)."""
    head, tail = name.rsplit(".", 1)
    return f"{head}.{SHARED_INPUT_LEADER[tail]}" if tail in SHARED_INPUT_LEADER else name


@torch.no_grad()
def collect_hessians(model, calib_ids, target_names):
    """Sum of x^T x per LEADER module and token count, over the calibration windows (same loop as
    weight_gptq.collect_hessians_batch, but tolerant of experts that get no tokens)."""
    modules = dict(model.named_modules())
    leaders = sorted({leader_name(n) for n in target_names})
    h_cpu, count = {}, {}

    def make_hook(name):
        def hook(_mod, inputs):
            x2 = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(torch.float32)
            h = x2.t().matmul(x2).cpu()
            if name not in h_cpu:
                h_cpu[name], count[name] = h, int(x2.shape[0])
            else:
                h_cpu[name].add_(h)
                count[name] += int(x2.shape[0])
        return hook

    handles = [modules[n].register_forward_pre_hook(make_hook(n)) for n in leaders]
    try:
        for begin in range(0, calib_ids.shape[1], GPTQ_CALIB_STRIDE):
            end = min(begin + GPTQ_CALIB_SEQ_LEN, calib_ids.shape[1])
            if end <= begin:
                break
            model(input_ids=calib_ids[:, begin:end].to(model.device), use_cache=False)
            if (begin // GPTQ_CALIB_STRIDE) + 1 >= GPTQ_CALIB_STEPS:
                break
    finally:
        for h in handles:
            h.remove()
    return h_cpu, count


def build_cache(fmt: str, calib_ids: torch.Tensor, model=None, n_devices=None, procs_per_device=None,
                rotate_group: int = 0, loader=None, prepare_model=None) -> dict:
    """rotate_group > 0: micro-rotated GPTQ (see common/micro_rotation.py) -- the returned cache holds the
    quantized ROTATED weights. loader: () -> model, for other models whose Linears are plain nn.Linear
    (e.g. Llama); the model is dropped before the workers start. rotate_group == -1: full rotation.
    prepare_model(model): applied right after loading (norm folding), so Hessians and weights come from it."""
    model = model if model is not None else (loader() if loader is not None else dsm.load_model())
    if prepare_model is not None:  # e.g. RMSNorm folding for the full (QuaRot-style) rotation; before Hessians/weights
        prepare_model(model)
    n_devices = n_devices or torch.cuda.device_count()
    procs_per_device = procs_per_device or int(os.environ.get("GPTQ_PROCS_PER_DEVICE", "6"))
    target_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n != "lm_head"]
    print(f"[{fmt}] {len(target_names)} target Linear modules", flush=True)

    t0 = time.time()
    h_cpu, count = collect_hessians(model, calib_ids, target_names)
    print(f"[{fmt}] Hessians collected for {len(h_cpu)} input groups in {time.time() - t0:.0f}s", flush=True)

    modules = dict(model.named_modules())
    weights = {n: modules[n].weight.detach().cpu() for n in target_names}  # bf16, ~33GB host RAM
    del modules, model
    cleanup_cuda()

    n_fallback = {"n": 0}

    def jobs():
        for n in target_names:
            w = weights.pop(n)
            lead = leader_name(n)
            if lead in h_cpu and count[lead] > 0:
                h = h_cpu[lead] / float(count[lead])
            else:
                h = torch.eye(w.shape[1], dtype=torch.float32)  # expert with no calibration tokens
                n_fallback["n"] += 1
            yield (n, w, h.contiguous())
            # drop a Hessian once its last sharer (v_proj / up_proj) or its only user has been dispatched;
            # q_proj and gate_proj are followed by sharers, so they must keep it
            if n.rsplit(".", 1)[-1] not in ("q_proj", "gate_proj"):
                h_cpu.pop(lead, None)

    t0 = time.time()
    cache, rels = run_jobs(jobs(), fmt, n_devices, procs_per_device, rotate_group=rotate_group)
    r = torch.tensor(list(rels.values()))
    print(f"[{fmt}] quantized {len(cache)} matrices in {time.time() - t0:.0f}s "
          f"({n_fallback['n']} used the identity-Hessian fallback); relative weight MSE: "
          f"mean={r.mean():.4f} median={r.median():.4f} max={r.max():.4f}", flush=True)
    return cache



@torch.no_grad()
def measure_weight_only(fmt: str, cache: dict, encodings) -> dict:
    """BF16 vs the GPTQ weight cache loaded onto an otherwise-untouched (activations stay BF16)
    model -- mirrors weight_gptq.measure_weight_only / weight_gptq_qwen3_moe.measure_weight_only
    for this model's own loader (dsm.load_model, accelerate device_map="auto")."""
    same_seeds()
    model = dsm.load_model()
    bf16_ppl = measure_ppl(model, encodings, "deepseek_moe_16b__BF16", model.device)
    del model
    cleanup_cuda()

    same_seeds()
    model = dsm.load_model()
    modules = dict(model.named_modules())
    n = 0
    for name, w in cache.items():
        modules[name].weight.data.copy_(w.to(modules[name].weight.device, modules[name].weight.dtype))
        n += 1
    weight_ppl = measure_ppl(model, encodings, "deepseek_moe_16b__Weight_ONLY", model.device)
    del model
    cleanup_cuda()

    print(f"[{fmt}][deepseek_moe_16b] BF16={bf16_ppl:.6f}  Weight_ONLY={weight_ppl:.6f} "
          f"(+{weight_ppl - bf16_ppl:.4f})  n_layers={n}", flush=True)
    return {"format": fmt, "model": "deepseek_moe_16b", "bf16_ppl": bf16_ppl, "weight_only_ppl": weight_ppl,
            "increase": weight_ppl - bf16_ppl, "n_layers": n}


def main():
    fmt = sys.argv[1]
    path = cache_path_for(fmt)
    same_seeds()
    if os.path.exists(path):
        print(f"[{fmt}] cache already exists -> {path} (loading, not rebuilding)", flush=True)
        cache = torch.load(path, map_location="cpu")
    else:
        _, calib_encodings = load_tokenizer_and_encodings(dsm.MODEL_NAME, load_wikitext2_calib_text())
        t0 = time.time()
        cache = build_cache(fmt, calib_encodings.input_ids)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        torch.save(cache, tmp)
        os.replace(tmp, path)
        print(f"[{fmt}] cache saved -> {path} ({time.time() - t0:.1f}s)", flush=True)

    if os.environ.get("GPTQ_SKIP_PPL", "0") == "1":
        del cache
        cleanup_cuda()
        return

    _, encodings = load_tokenizer_and_encodings(dsm.MODEL_NAME, load_wikitext2_text())
    row = measure_weight_only(fmt, cache, encodings)
    del cache
    cleanup_cuda()

    csv_path = os.path.join(RESULTS_DIR, f"weight_gptq_ppl_{fmt}_deepseek_moe_16b.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["format", "model", "bf16_ppl", "weight_only_ppl", "increase", "n_layers"])
        w.writeheader()
        w.writerow(row)
    print(f"Saved -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
