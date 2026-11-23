"""Figure 5 (MoE models, Qwen3-30B-A3B and DeepSeek-MoE-16B-Base): same mantissa-bitwidth sweep as dense.py
(reduction_pct_all_m, accumulate, FORMAT_CONFIG imported from it).

MoE expert inputs are probed on every active expert call, so each expert only contributes the tokens
routed to it:
  qwen3_moe        : q/k/v/o_proj are nn.Linear (forward-pre-hooks). Experts store gate_up/down as
                     batched Parameters, probed inline via common/qwen3_moe_targets.patch_qwen3_moe_experts.
                     Q/K are probed after QK-Norm and RoPE (patch_eager_qwen3_moe).
  deepseek_moe_16b : common/deepseek_moe_model.py (pure-PyTorch port). Every routed expert, the shared
                     experts and the dense layer-0 MLP are nn.Linear; gate_proj ("gateup_proj"),
                     down_proj, q_proj ("qkv_proj") and o_proj are hooked. Q/K via set_hooks after RoPE.

Usage: CUDA_VISIBLE_DEVICES=<free GPUs> python3 moe.py <nvfp4|mxfp4> <qwen3_moe|deepseek_moe_16b>
Output: results/real_summary_{fmt}_{model}.csv
"""
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_tokenizer_and_encodings  # noqa: E402
from dense import (FORMAT_CONFIG, PROBE_SEQ_LEN, RESULTS_DIR, reduction_pct_all_m, accumulate,  # noqa: E402
                   average, write_summary, identity)

# Linear module-name suffix -> category, per model (kept as in the measured runs)
LINEAR_CATEGORY = {
    "qwen3_moe": {"q_proj": "qkv_proj", "k_proj": "qkv_proj", "v_proj": "qkv_proj", "o_proj": "o_proj"},
    "deepseek_moe_16b": {"q_proj": "qkv_proj", "o_proj": "o_proj", "gate_proj": "gateup_proj", "down_proj": "down_proj"},
}


def model_name_of(model_key):
    if model_key == "qwen3_moe":
        from qwen3_moe_targets import MODEL_NAME
    else:
        from deepseek_moe_model import MODEL_NAME
    return MODEL_NAME


def build_model(model_key, probe):
    """Returns (model, restore). Patches are applied BEFORE loading (accelerate captures forwards at load time)."""
    if model_key == "qwen3_moe":
        from qwen3_moe_targets import MODEL_NAME, patch_eager_qwen3_moe, patch_qwen3_moe_experts
        import transformers.models.qwen3_moe.modeling_qwen3_moe as qwen3_mod
        attn_cls, orig_attn = patch_eager_qwen3_moe(qwen3_mod, probe("Q"), probe("K"), identity, identity)
        experts_cls, orig_experts = patch_qwen3_moe_experts(qwen3_mod, probe("gateup_proj"), probe("down_proj"))
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                       attn_implementation="eager", device_map="auto")
        restore = lambda: (setattr(attn_cls, "forward", orig_attn), setattr(experts_cls, "forward", orig_experts))
    else:
        import deepseek_moe_model as dsm
        dsm.set_hooks(q=probe("Q"), k=probe("K"))
        model = dsm.load_model()
        restore = dsm.set_hooks
    model.eval()
    return model, restore


@torch.no_grad()
def measure_model(fmt: str, model_key: str) -> dict:
    cfg = FORMAT_CONFIG[fmt]
    k, block_size, base_config = cfg["k"], cfg["block_size"], cfg["base_config"]

    same_seeds()
    _, encodings = load_tokenizer_and_encodings(model_name_of(model_key), load_wikitext2_text())

    cat_sum, cat_n = {}, {}

    def probe(category):
        def fn(x):
            accumulate(cat_sum, cat_n, category, reduction_pct_all_m(x.detach(), fmt, k, block_size, base_config))
            return x
        return fn

    model, restore = build_model(model_key, probe)

    def make_hook(category):
        p = probe(category)

        def hook(_mod, inputs):  # returns None: the Linear input is left unchanged
            p(inputs[0])
        return hook

    handles = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or name == "lm_head" or name.endswith(".lm_head"):
            continue
        category = LINEAR_CATEGORY[model_key].get(name.rsplit(".", 1)[-1])
        if category is not None:
            handles.append(mod.register_forward_pre_hook(make_hook(category)))
    print(f"Hooking {len(handles)} Linear modules", flush=True)

    model(input_ids=encodings.input_ids[:, :PROBE_SEQ_LEN].to(model.device))
    restore()
    for h in handles:
        h.remove()

    by_category = average(fmt, model_key, cat_sum, cat_n)
    del model
    cleanup_cuda()
    return by_category


def main():
    fmt, model_key = sys.argv[1], sys.argv[2]
    by_category = measure_model(fmt, model_key)
    write_summary(os.path.join(RESULTS_DIR, f"real_summary_{fmt}_{model_key}.csv"), {model_key: by_category})
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
