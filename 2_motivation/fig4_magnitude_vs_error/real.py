"""H (no number, Methodology slide): does the block's magnitude-max
element coincide with the element carrying the largest quantization
error -- and how much of a block's total squared error does the
magnitude-max element alone carry? This is the direct premise check
behind block-max-based boosting (MX+'s BM, and this project's own
motivation): if the block max usually IS (or dominates) the largest-
error position, boosting it specifically is well justified.

Methodology (see slides):
  - Models: Llama-3.1-8B (base) here; Qwen3-30B-A3B and DeepSeek-MoE-16B in real_moe.py
  - Dataset: WikiText-2, one 2048-token sequence
  - Collect BF16 X, Q, K activations from all Transformer layers in a
    single forward pass (X = every target Linear's input, pooled
    across ALL Linear categories into one target -- unlike H2/H4's
    per-category split).
  - Quantize each collected activation block to NVFP4 and MXFP4,
    compute element-wise squared quantization error eps_i^2 = (a_i-a_hat_i)^2.
  - Per block: match rate = 1[argmax_i|a_i| == argmax_i eps_i^2];
    block-max error contribution = eps_{argmax|a|}^2 / sum_i eps_i^2.
  - Average each metric across blocks and layers, per activation
    target (X/Q/K), report NVFP4 and MXFP4 separately.

Usage: python3 real.py [model ...]
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

    # fc2 (OPT's MLP down-projection) is reported separately from the
    # pooled "X" target -- ReLU makes its input extremely sparse
    # (block-max is frequently the only nonzero element), which otherwise
    # skews the pooled X average for OPT specifically (llama/mistral have
    # no fc2 at all, so this only ever fires for OPT).
    def make_x_hook(target):
        def x_hook(_mod, inputs):
            x = inputs[0].detach()
            for fmt, cfg in FORMAT_CONFIG.items():
                accumulate(acc, target, fmt, block_stats(x, cfg["block_size"], cfg["base_config"]))
        return x_hook

    handles = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and not should_skip_linear(n):
            target = "X_fc2" if n.endswith("fc2") else "X"
            handles.append(m.register_forward_pre_hook(make_x_hook(target)))

    def q_probe(x):
        for fmt, cfg in FORMAT_CONFIG.items():
            accumulate(acc, "Q", fmt, block_stats(x.detach(), cfg["block_size"], cfg["base_config"]))
        return x

    def k_probe(x):
        for fmt, cfg in FORMAT_CONFIG.items():
            accumulate(acc, "K", fmt, block_stats(x.detach(), cfg["block_size"], cfg["base_config"]))
        return x

    if arch == "opt":
        import transformers.models.opt.modeling_opt as attn_mod
    elif "ministral" in model_name.lower():
        import transformers.models.ministral.modeling_ministral as attn_mod
    elif "mistral" in model_name.lower():
        import transformers.models.mistral.modeling_mistral as attn_mod
    else:
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
