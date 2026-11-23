"""Weight GPTQ + Hessian-weighted scaling-factor search for Qwen3-30B-A3B's
MoE experts (gate_up_proj/down_proj) -- extends weight_gptq.py's per-Linear
GPTQ to the batched-parameter expert structure (see common/qwen3_moe_targets.py's
module docstring for why gate_up_proj/down_proj aren't nn.Linear here: they're
raw nn.Parameter tensors of shape [num_experts, ...], matmul'd per active
expert via nn.functional.linear inside a Python loop, not one nn.Linear
per expert).

Q/K/V/O_proj ARE plain nn.Linear (confirmed in qwen3_moe_targets.py), so
those go through the EXACT SAME iter_target_linear_names/collect_hessian/
gptq_quantize_weight pipeline as weight_gptq.py, completely unmodified --
only imported and reused here.

For gate_up_proj/down_proj: patches Qwen3MoeExperts.forward (same per-
active-expert loop as qwen3_moe_targets.patch_qwen3_moe_experts) to
accumulate a PER-(layer, expert) Hessian (X^T X, only over the tokens
actually routed to that expert during calibration) instead of a probe/
error metric. gptq_quantize_weight (imported from weight_gptq.py
UNCHANGED, including its ill-conditioning guard -- see that function's
docstring) is then called once per expert per matrix, exactly like it's
called once per dense Linear layer elsewhere -- the algorithm doesn't
know or care that its 2D weight came from an expert slice instead of a
standalone nn.Linear.

Memory: with 48 layers x 128 experts x 2 Hessians (2048x2048 for
gate_up_proj's input dim, 768x768 for down_proj's), holding every
expert's Hessian simultaneously during ONE calibration pass costs ~118GB
of host RAM at peak (checked against this box's 237GB available before
choosing this over re-running calibration once per layer, which would
need 48x the forward-pass compute for no accuracy benefit). Freed
incrementally as each expert is quantized and consumed.

Calibration coverage caveat: with the limited calibration budget (8 steps
x 512 tokens = 4096 tokens x top-8 routing = 32768 token-expert
assignments per layer across 128 experts, ~256 tokens/expert on average),
most experts get calibrated, but a rare expert can end up with zero
routed tokens in this window. Those fall back to gptq_quantize_weight
called with an IDENTITY Hessian instead of skipping the expert or leaving
it BF16 -- with h=I, diag is uniform (unweighted per-block MSE search,
same as Ingredient 1 minus Hessian-weighting), and the OBS correction
term solve(h_bb, h_br) is ~0 (h_br is ~0 off the diagonal for identity),
so this reduces to a plain, independent, best-effort per-block quantize
with no risk of the cross-block instability this project's condition-
number guard was written for -- no special-casing needed, the existing
function already degrades gracefully.

Device: gate_up_proj/down_proj require device_map="auto" across the free
GPUs (30.5B total / 3.3B active doesn't fit one 24GB card, same as the
activation-side scripts) -- the experts-forward patch MUST be applied BEFORE
from_pretrained (accelerate captures each module's forward at load time).

Cache: $PTQ_CACHE_ROOT/{fmt}/qwen3_moe/ptq_cache_NVFP4.pt, keyed by Linear name for Q/K/V/O and by full
parameter name (e.g. "model.layers.5.mlp.experts.gate_up_proj", the whole [128, ...] tensor) for the two
expert tensors; common/ppl_w4a4.load_cache_into handles both.

Usage: GPTQ_CPU=0 CUDA_VISIBLE_DEVICES=<free GPUs only> python3 weight_gptq_qwen3_moe.py <nvfp4|mxfp4>
"""
import csv
import os
import sys
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gptq_parallel import run_jobs
from weight_gptq import (
    DEVICE, GPTQ_CPU, FORMAT_CONFIG, GPTQ_CALIB_SEQ_LEN, GPTQ_CALIB_STRIDE, GPTQ_CALIB_STEPS,
    should_skip_linear, gptq_quantize_weight, collect_hessians_batch,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_wikitext2_calib_text, load_tokenizer_and_encodings, measure_ppl
from formats import quantize_tensor_config
from quant_targets import replace_linear_modules
from qwen3_moe_targets import MODEL_NAME, patch_eager_qwen3_moe, patch_qwen3_moe_experts

torch.manual_seed(1111)

PTQ_CACHE_ROOT = os.environ.get("PTQ_CACHE_ROOT", os.path.expanduser("~/ptq_cache"))
def cache_path_for(fmt: str) -> str:
    return os.path.join(PTQ_CACHE_ROOT, fmt, "qwen3_moe", "ptq_cache_NVFP4.pt")


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    m = model
    for part in name.split("."):
        m = getattr(m, part)
    return m


def iter_target_linear_names(model: nn.Module):
    """Q/K/V/O_proj only -- gate_up_proj/down_proj are handled separately below."""
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and not should_skip_linear(n)]


def patch_qwen3_moe_experts_calib(mod):
    """Reimplements Qwen3MoeExperts.forward's per-active-expert loop,
    accumulating a Hessian (X^T X) per (module instance, expert_idx, "gateup"
    or "down") instead of doing the real matmul against gate_up_proj/
    down_proj -- SAME loop shape as qwen3_moe_targets.patch_qwen3_moe_experts,
    diverged here because that one probes an error metric and this one
    needs to keep the running Hessian sum + token count across multiple
    calibration forward passes (call sites accumulate into a dict shared
    across the whole calibration run, not reset per call)."""
    cls = mod.Qwen3MoeExperts
    orig_forward = cls.forward
    hessians: dict = {}  # (id(self), expert_idx, kind) -> [H_sum(cpu), token_count]

    def accumulate(key, x):
        x2 = x.detach().reshape(-1, x.shape[-1]).to(torch.float32)
        h = x2.cpu().t().matmul(x2.cpu()) if GPTQ_CPU else x2.t().matmul(x2).cpu()
        if key not in hessians:
            hessians[key] = [h, int(x2.shape[0])]
        else:
            hessians[key][0].add_(h)
            hessians[key][1] += int(x2.shape[0])

    def patched_forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            accumulate((id(self), int(expert_idx), "gateup"), current_state)
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            accumulate((id(self), int(expert_idx), "down"), current_hidden_states)
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    cls.forward = patched_forward
    return cls, orig_forward, hessians


@torch.no_grad()
def build_cache(fmt: str, calib_ids: torch.Tensor) -> dict:
    import transformers.models.qwen3_moe.modeling_qwen3_moe as attn_mod
    experts_cls, orig_experts_forward, hessians = patch_qwen3_moe_experts_calib(attn_mod)

    from transformers import AutoModelForCausalLM
    kw = {}
    # On a box with few/no spare GPUs, device_map="auto" with no max_memory crams the whole
    # ~60GB bf16 model onto the visible GPU(s) first, leaving no headroom for norm-folding /
    # Hessian-collection activations (real OOM: 22.6GiB of 23.52GiB used by weights alone on a
    # single-GPU box). GPTQ_MAX_MEM_PER_GPU caps each visible GPU's weight budget so the rest
    # spills to host RAM via accelerate's own CPU fallback.
    per_gpu = os.environ.get("GPTQ_MAX_MEM_PER_GPU")
    if per_gpu:
        kw["max_memory"] = {i: per_gpu for i in range(torch.cuda.device_count())}
        kw["max_memory"]["cpu"] = os.environ.get("GPTQ_MAX_MEM_CPU", "200GiB")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                                   low_cpu_mem_usage=True, attn_implementation="eager",
                                                   device_map="auto", **kw)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # id(module) -> (layer_name, module) so accumulated Hessians (keyed by id(self)) can be
    # traced back to a name after the patched forward has already run and returned.
    experts_by_id = {id(m): (n, m) for n, m in model.named_modules() if isinstance(m, attn_mod.Qwen3MoeExperts)}
    print(f"found {len(experts_by_id)} MoE layers", flush=True)

    # ---- Q/K/V/O_proj + MoE experts: ONE calibration pass collects both.
    # Q/K/V/O Hessians used to be gathered via weight_gptq.collect_hessian
    # called once PER LAYER (192 layers x its own GPTQ_CALIB_STEPS-step
    # forward loop = up to 1536 redundant full-model forward passes through
    # this 3-GPU-sharded 30B model -- confirmed via wall-clock/nvidia-smi to
    # be the actual bottleneck, not GPU underutilization, when a real run
    # crawled through only 32/192 layers in over an hour). collect_hessians_
    # batch hooks all 192 Linears simultaneously and shares this same
    # forward loop with the expert-Hessian patch already installed above.
    linear_names = iter_target_linear_names(model)
    linear_modules = {n: get_module_by_name(model, n) for n in linear_names}
    print(f"collecting Hessians for {len(linear_names)} Q/K/V/O linears + MoE experts "
          f"in a single calibration pass...", flush=True)
    linear_hessians = collect_hessians_batch(model, linear_modules, calib_ids)
    experts_cls.forward = orig_experts_forward
    print(f"collected Hessians for {len(hessians)} (layer, expert, kind) entries "
          f"(of up to {len(experts_by_id) * 128 * 2})", flush=True)

    # ---- Quantization phase: gptq_quantize_weight (UNCHANGED) for every matrix, run through
    # gptq_parallel.run_jobs -- several worker processes per GPU. One matrix is latency-bound (~3-5 s
    # NVFP4 on an otherwise idle GPU) and the per-device threads used before only gave ~1.2-1.4x (GIL);
    # only WHEN each independent matrix runs changes, not the GPTQ/OBS math or the scale search.
    # Weights are copied to host RAM first so the model can be freed: the workers need the GPU memory.
    def _real(module, attr):
        # Under device_map="auto" + max_memory CPU overflow, accelerate's AlignDevicesHook
        # replaces offloaded params with a meta placeholder outside of a forward pass; the
        # actual CPU tensor (post norm-folding) lives in the hook's weights_map instead.
        t = getattr(module, attr)
        if t.device.type != "meta":
            return t.detach().cpu()
        return module._hf_hook.weights_map[attr].detach().cpu()

    linear_weights = {n: _real(linear_modules[n], "weight") for n in linear_names}
    expert_layers = {name: (id(m), int(m.num_experts), _real(m, "gate_up_proj"), _real(m, "down_proj"))
                     for name, m in experts_by_id.values()}
    del linear_modules, experts_by_id, model
    cleanup_cuda()

    n_fallback = {"n": 0}

    def expert_hessian(key, in_dim):
        if key in hessians:
            h_sum, count = hessians.pop(key)
            return (h_sum / float(count)).contiguous()
        n_fallback["n"] += 1  # expert got no calibration tokens -> identity Hessian (see module docstring)
        return torch.eye(in_dim, dtype=torch.float32)

    def jobs():
        for name in linear_names:
            yield (name, linear_weights.pop(name), linear_hessians.pop(name).contiguous())
        # pop() per layer as its jobs are generated (not after ALL layers finish quantizing):
        # keeps only the current layer's raw bf16 weights resident while `results` (the
        # growing quantized-output dict) fills up, instead of holding every layer's raw copy
        # alive for the whole run -- this was doubling peak host RAM right as quantization
        # finished (full raw expert_layers + full quantized results coexisting).
        for lname in list(expert_layers):
            mid, n_exp, gu, dn = expert_layers.pop(lname)
            for e in range(n_exp):
                # .clone(): a view would drag the whole [E, ...] storage through the queue
                w = gu[e].clone()
                yield (f"{lname}.gate_up_proj#{e}", w, expert_hessian((mid, e, "gateup"), w.shape[1]))
                w = dn[e].clone()
                yield (f"{lname}.down_proj#{e}", w, expert_hessian((mid, e, "down"), w.shape[1]))

    n_devices = max(1, torch.cuda.device_count())
    procs_per_device = int(os.environ.get("GPTQ_PROCS_PER_DEVICE", "4"))
    print(f"quantizing {len(linear_names)} Q/K/V/O Linears + {len(expert_layers)} MoE layers x "
          f"{next(iter(expert_layers.values()))[1]} experts x 2 matrices on {n_devices} GPUs x {procs_per_device} procs", flush=True)
    # shapes/dtypes only (not the tensors themselves) for reassembling cache[...] after the run --
    # expert_layers itself is popped empty by jobs() below as quantization proceeds.
    layer_meta = {lname: (n_exp, gu.shape, gu.dtype, dn.shape, dn.dtype)
                  for lname, (mid, n_exp, gu, dn) in expert_layers.items()}
    t0 = time.time()
    results, rels = run_jobs(jobs(), fmt, n_devices, procs_per_device)
    r = torch.tensor(list(rels.values()))
    print(f"quantized {len(results)} matrices in {time.time() - t0:.0f}s ({n_fallback['n']} expert matrices used the "
          f"identity-Hessian fallback); relative weight MSE: mean={r.mean():.4f} median={r.median():.4f} max={r.max():.4f}", flush=True)

    cache: dict = {}
    for name in linear_names:
        cache[name] = results.pop(name)
    for lname, (n_exp, gu_shape, gu_dtype, dn_shape, dn_dtype) in layer_meta.items():
        gu_out = torch.empty(gu_shape, dtype=gu_dtype)
        dn_out = torch.empty(dn_shape, dtype=dn_dtype)
        for e in range(n_exp):
            gu_out[e] = results.pop(f"{lname}.gate_up_proj#{e}")
            dn_out[e] = results.pop(f"{lname}.down_proj#{e}")
        cache[f"{lname}.gate_up_proj"] = gu_out
        cache[f"{lname}.down_proj"] = dn_out
    assert not results, f"unassembled results left: {list(results)[:3]}"
    return cache


def main():
    fmt = sys.argv[1]
    path = cache_path_for(fmt)
    if os.path.exists(path):
        print(f"[{fmt}][qwen3_moe] cache already exists -> {path}", flush=True)
        return
    same_seeds()
    _, calib_encodings = load_tokenizer_and_encodings(MODEL_NAME, load_wikitext2_calib_text())
    t0 = time.time()
    cache = build_cache(fmt, calib_encodings.input_ids)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(cache, path + ".tmp")
    os.replace(path + ".tmp", path)
    print(f"[{fmt}][qwen3_moe] cache saved -> {path} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
