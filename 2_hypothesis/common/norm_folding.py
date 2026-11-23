"""RMSNorm folding (the first QuaRot step): the per-channel norm weight gamma is folded into the Linear
layers that read the normalized hidden state, and the norm weight is set to 1.

    y = W (gamma * xhat) = (W diag(gamma)) xhat        xhat = x / rms(x)

Afterwards the quantized activation at those Linears is xhat (rotated), not gamma * xhat, which is what QuaRot
quantizes; without folding the rotation would act on a differently scaled tensor. Consumers of each norm:
  input_layernorm          -> self_attn.q_proj / k_proj / v_proj
  post_attention_layernorm -> every gate_proj / up_proj under mlp (dense, shared and routed experts), the router
                              (mlp.gate.weight), and batched expert weights (mlp.experts.gate_up_proj)
  model.norm               -> lm_head
Weights are multiplied along their input (last) dimension in fp32 and cast back to their own dtype.
Structure-based: works for the Llama, Qwen3-MoE and DeepSeek-MoE (pure-PyTorch port) modules of this project.
"""
import torch


def _fold_into(tensor: torch.Tensor, gamma: torch.Tensor) -> None:
    tensor.data.copy_((tensor.data.float() * gamma.float().to(tensor.device)).to(tensor.dtype))


@torch.no_grad()
def fold_norms(model) -> int:
    if getattr(model, "_norms_folded", False):
        raise RuntimeError("norms already folded")
    n = 0
    for layer in model.model.layers:
        g_in = layer.input_layernorm.weight.data.clone()
        for name in ("q_proj", "k_proj", "v_proj"):
            _fold_into(getattr(layer.self_attn, name).weight, g_in)
            n += 1
        layer.input_layernorm.weight.data.fill_(1.0)

        g_post = layer.post_attention_layernorm.weight.data.clone()
        for pname, p in layer.mlp.named_parameters():
            if (pname.endswith("gate_proj.weight") or pname.endswith("up_proj.weight")
                    or pname == "gate.weight" or pname.endswith("experts.gate_up_proj")):
                _fold_into(p, g_post)
                n += 1
        layer.post_attention_layernorm.weight.data.fill_(1.0)

    g_final = model.model.norm.weight.data.clone()
    _fold_into(model.lm_head.weight, g_final)
    model.model.norm.weight.data.fill_(1.0)
    model._norms_folded = True
    return n + 1
