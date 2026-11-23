"""H1 (block-max vs largest-error premise check) for the MoE models: Qwen3-30B-A3B and
DeepSeek-MoE-16B-Base. Same metrics/pooling as real.py (imported from it): X = every Linear
input pooled into ONE target (gate and up counted as two Linears that share one input, like
the dense script's gate_proj/up_proj), plus Q and K after RoPE; NVFP4 and MXFP4 computed
in one forward pass over one 2048-token WikiText-2 sequence.

  qwen3_moe        : attention Linears are plain nn.Linear (hooked); expert gate_up/down are batched
                     Parameters -> probed inline via common/qwen3_moe_targets.patch_qwen3_moe_experts
                     (the fused gate_up matmul input is counted twice = gate + up).
  deepseek_moe_16b : common/deepseek_moe_model.py (pure-PyTorch port); every Linear (attention,
                     all routed experts, shared experts, dense layer 0) is hooked.

Usage: CUDA_VISIBLE_DEVICES=<free GPUs> python3 real_moe.py <qwen3_moe|deepseek_moe_16b>
"""
import csv
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_tokenizer_and_encodings
from quant_targets import should_skip_linear
from real import block_stats, accumulate, FORMAT_CONFIG, PROBE_SEQ_LEN, RESULTS_DIR, identity

FIELDS = ["model", "target", "format", "match_rate", "blockmax_error_contribution", "largest_error_contribution", "n_blocks"]


@torch.no_grad()
def measure_model(model_key: str) -> dict:
    same_seeds()
    full_text = load_wikitext2_text()
    acc = {}

    def probe(target):
        def fn(x):
            for fmt, cfg in FORMAT_CONFIG.items():
                accumulate(acc, target, fmt, block_stats(x.detach(), cfg["block_size"], cfg["base_config"]))
            return x
        return fn

    x_probe, q_probe, k_probe = probe("X"), probe("Q"), probe("K")

    if model_key == "qwen3_moe":
        from qwen3_moe_targets import MODEL_NAME, patch_eager_qwen3_moe, patch_qwen3_moe_experts
        import transformers.models.qwen3_moe.modeling_qwen3_moe as attn_mod
        _, encodings = load_tokenizer_and_encodings(MODEL_NAME, full_text)

        def gateup_probe(x):          # fused gate_up matmul == gate_proj + up_proj on one shared input
            x_probe(x)
            return x_probe(x)

        attn_cls, orig_attn = patch_eager_qwen3_moe(attn_mod, q_probe, k_probe, identity, identity)
        experts_cls, orig_experts = patch_qwen3_moe_experts(attn_mod, gateup_probe, x_probe)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                       attn_implementation="eager", device_map="auto")
        restore = lambda: (setattr(attn_cls, "forward", orig_attn), setattr(experts_cls, "forward", orig_experts))
    elif model_key == "deepseek_moe_16b":
        import deepseek_moe_model as dsm
        _, encodings = load_tokenizer_and_encodings(dsm.MODEL_NAME, full_text)
        dsm.set_hooks(q=q_probe, k=k_probe)
        model = dsm.load_model()
        restore = dsm.set_hooks
    else:
        raise SystemExit(f"unknown model {model_key}")
    model.eval()

    handles = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and not should_skip_linear(n):
            handles.append(m.register_forward_pre_hook(lambda _m, inputs: x_probe(inputs[0])))
    print(f"[{model_key}] hooked {len(handles)} Linear modules", flush=True)

    ids = encodings.input_ids[:, :PROBE_SEQ_LEN].to(model.device)
    model(input_ids=ids)
    restore()
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
    mk = sys.argv[1]
    res = measure_model(mk)
    csv_path = os.path.join(RESULTS_DIR, f"real_{mk}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for (target, fmt), s in res.items():
            w.writerow(dict(model=mk, target=target, format=fmt, match_rate=s["match_rate"],
                            blockmax_error_contribution=s["contribution"],
                            largest_error_contribution=s["largest_err_contribution"], n_blocks=s["n"]))
    print(f"Saved -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
