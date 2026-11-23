"""Figure 4 (dense model, Llama-3.1-8B): quantization error of the largest-magnitude element vs the
largest-error element in each NVFP4 / MXFP4 block.

Per block (eps_i^2 = (a_i - Q(a)_i)^2):
  match_rate                  : 1[argmax_i |a_i| == argmax_i eps_i^2]
  blockmax_error_contribution : eps_{argmax|a|}^2 / sum_i eps_i^2   (largest-magnitude element)
  largest_error_contribution  : max_i eps_i^2 / sum_i eps_i^2       (largest-error element)
averaged over all blocks of each target. Figure 4's normalized squared error of the largest-magnitude
element = blockmax_error_contribution / largest_error_contribution (largest-error element = 1).

Formats: NVFP4 = E2M1_amax_E4M3 (block 16), MXFP4 = E2M1_mxfp4_ue8m0 (block 32), both in one pass.
Targets: X = every Linear input pooled into one target, Q and K after RoPE.
Input: one 2048-token WikiText-2 sequence.
MoE models (Qwen3-30B-A3B, DeepSeek-MoE-16B): moe.py. Laplace model: laplace.py.
merge_results.py combines the per-model results/real_{model}.csv into results/real_summary.csv.

Usage: python3 dense.py [model_key ...]   (default: llama_base)
Output: results/real_{model}.csv
"""
import csv
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_tokenizer_and_encodings
from formats import quantize_tensor_config
from quant_targets import should_skip_linear, patch_eager, MODEL_CONFIG

torch.manual_seed(1111)

FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, base_config="E2M1_amax_E4M3"),
    "mxfp4": dict(block_size=32, base_config="E2M1_mxfp4_ue8m0"),
}
PROBE_SEQ_LEN = 2048

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


@torch.no_grad()
def block_stats(x: torch.Tensor, block_size: int, base_config: str) -> dict:
    """x: [..., K]. Returns per-format-config match/contribution sums (to
    be accumulated across all layers) plus the valid block count."""
    K = x.shape[-1]
    n_blocks = K // block_size
    if n_blocks == 0:
        return None
    x = x.reshape(-1, K)[:, : n_blocks * block_size].to(torch.float32)
    xq = quantize_tensor_config(x, base_config, block_size)
    eps_sq = (x - xq).pow(2).reshape(-1, block_size)
    blocks = x.reshape(-1, block_size)

    max_idx = blocks.abs().argmax(dim=-1)
    err_idx = eps_sq.argmax(dim=-1)
    total_err = eps_sq.sum(dim=-1)
    valid = total_err > 0
    if valid.sum() == 0:
        return None

    match = (max_idx == err_idx)[valid].float().sum().item()
    blockmax_err = eps_sq.gather(-1, max_idx.unsqueeze(-1)).squeeze(-1)
    contribution = (blockmax_err[valid] / total_err[valid].clamp(min=1e-30)).sum().item()
    # largest-error element's OWN contribution (the ceiling -- equals
    # blockmax's contribution exactly on blocks where match=1, and is
    # strictly larger elsewhere since eps_sq.max() >= blockmax_err by
    # definition of err_idx).
    largest_err_err = eps_sq.gather(-1, err_idx.unsqueeze(-1)).squeeze(-1)
    largest_err_contribution = (largest_err_err[valid] / total_err[valid].clamp(min=1e-30)).sum().item()
    n_valid = int(valid.sum().item())
    return dict(match=match, contribution=contribution, largest_err_contribution=largest_err_contribution, n=n_valid)


def accumulate(acc: dict, target: str, fmt: str, res: dict):
    if res is None:
        return
    key = (target, fmt)
    if key not in acc:
        acc[key] = dict(match=0.0, contribution=0.0, largest_err_contribution=0.0, n=0)
    acc[key]["match"] += res["match"]
    acc[key]["contribution"] += res["contribution"]
    acc[key]["largest_err_contribution"] += res["largest_err_contribution"]
    acc[key]["n"] += res["n"]


def identity(x):
    return x


@torch.no_grad()
def measure_model(model_key: str) -> dict:
    model_cfg = MODEL_CONFIG[model_key]
    model_name, arch = model_cfg["model_name"], model_cfg["arch"]
    same_seeds()
    full_text = load_wikitext2_text()
    _, encodings = load_tokenizer_and_encodings(model_name, full_text)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                   low_cpu_mem_usage=True, attn_implementation="eager")
    model.to(DEVICE)
    model.eval()

    acc = {}

    def x_hook(_mod, inputs):
        x = inputs[0].detach()
        for fmt, cfg in FORMAT_CONFIG.items():
            accumulate(acc, "X", fmt, block_stats(x, cfg["block_size"], cfg["base_config"]))

    handles = [m.register_forward_pre_hook(x_hook) for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and not should_skip_linear(n)]

    def q_probe(x):
        for fmt, cfg in FORMAT_CONFIG.items():
            accumulate(acc, "Q", fmt, block_stats(x.detach(), cfg["block_size"], cfg["base_config"]))
        return x

    def k_probe(x):
        for fmt, cfg in FORMAT_CONFIG.items():
            accumulate(acc, "K", fmt, block_stats(x.detach(), cfg["block_size"], cfg["base_config"]))
        return x

    import transformers.models.llama.modeling_llama as attn_mod
    attn_cls, orig_forward = patch_eager(arch, attn_mod, q_probe, k_probe, identity, identity)

    ids = encodings.input_ids[:, :PROBE_SEQ_LEN].to(DEVICE)
    model(input_ids=ids)  # single forward pass -- collects X/Q/K and computes both formats' block stats on the fly
    attn_cls.forward = orig_forward
    for h in handles:
        h.remove()

    results = {}
    for (target, fmt), s in acc.items():
        results[(target, fmt)] = dict(match_rate=100 * s["match"] / s["n"], contribution=100 * s["contribution"] / s["n"],
                                       largest_err_contribution=100 * s["largest_err_contribution"] / s["n"], n=s["n"])
        r = results[(target, fmt)]
        print(f"[{model_key}][{fmt}] {target}: match_rate={r['match_rate']:.2f}%  "
              f"blockmax_error_contribution={r['contribution']:.2f}%  "
              f"largest_error_contribution={r['largest_err_contribution']:.2f}%  (n={s['n']})", flush=True)

    del model
    cleanup_cuda()
    return results


def main():
    model_keys = sys.argv[1:] if len(sys.argv) > 1 else ["llama_base"]
    for mk in model_keys:
        res = measure_model(mk)
        rows = []
        for (target, fmt), s in res.items():
            rows.append(dict(model=mk, target=target, format=fmt, match_rate=s["match_rate"],
                              blockmax_error_contribution=s["contribution"],
                              largest_error_contribution=s["largest_err_contribution"], n_blocks=s["n"]))
        # per-model file -- avoids a shared-file race when models run in
        # parallel processes on different GPUs; merge_results.py combines them.
        csv_path = os.path.join(RESULTS_DIR, f"real_{mk}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "target", "format", "match_rate", "blockmax_error_contribution", "largest_error_contribution", "n_blocks"])
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"Saved -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
