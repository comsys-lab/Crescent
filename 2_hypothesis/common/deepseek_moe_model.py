"""Self-contained PyTorch port of deepseek-ai/deepseek-moe-16b-base (DeepseekForCausalLM).

Why a port: the checkpoint ships remote code (modeling_deepseek.py, written for
transformers 4.36) that does not even import on transformers 5.17
(is_torch_fx_available etc. are gone), and there is no native class for this
model_type ("deepseek", not deepseek_v2/v3). The architecture is small:
Llama-style MHA (16/16 heads, head_dim 128, no QK-norm, non-interleaved RoPE,
scale 1/sqrt(head_dim)) + layer 0 dense SwiGLU MLP + layers 1..27 MoE
(softmax router, top-6 of 64 routed experts, NO top-k renorm, plus 2 always-on
shared experts fused into one MLP). Parameter names match the checkpoint
exactly, so the safetensors load 1:1. Every expert is a plain nn.Linear triple,
so quant_targets.replace_linear_modules covers routed + shared experts (the
router is a bare F.linear on a Parameter and is not quantized, same as the
other MoE models here).

Per-surface quantization hooks: set_hooks(q, k, v, p) -- q/k/v act on
[B,H,S,D] after RoPE, p on the post-softmax probabilities (same positions as
every other patch in this project).
"""
import json
import math
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "deepseek-ai/deepseek-moe-16b-base"


def _identity(x):
    return x


HOOKS = dict(q=_identity, k=_identity, v=_identity, p=_identity)


def set_hooks(q=_identity, k=_identity, v=_identity, p=_identity):
    HOOKS.update(q=q, k=k, v=v, p=p)


# Error-compensation decomposition (same as quant_targets.patch_eager_ropegqa_decomposed /
# qwen3_moe_targets.patch_eager_qwen3_moe_decomposed): base_fns/boost_fns = {"q","k","v","p": fn}.
# QK^T = q_b k_B^T + q_B k_b^T - q_b k_b^T (drops the delta x delta term, like the hardware
# correction datapath); P@V likewise. None = plain single-precision path via HOOKS.
DECOMP = dict(fns=None)


def set_decomposed(base_fns=None, boost_fns=None):
    DECOMP["fns"] = None if base_fns is None else (base_fns, boost_fns)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class MLP(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.top_k = cfg["num_experts_per_tok"]
        self.norm_topk_prob = cfg["norm_topk_prob"]
        self.weight = nn.Parameter(torch.empty(cfg["n_routed_experts"], cfg["hidden_size"]))

    def forward(self, x):
        scores = F.linear(x.view(-1, x.shape[-1]), self.weight).softmax(dim=-1)
        w, idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.top_k > 1 and self.norm_topk_prob:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return idx, w


class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, mi = cfg["hidden_size"], cfg["moe_intermediate_size"]
        self.experts = nn.ModuleList(MLP(h, mi) for _ in range(cfg["n_routed_experts"]))
        self.gate = MoEGate(cfg)
        self.shared_experts = MLP(h, mi * cfg["n_shared_experts"])
        self.top_k = cfg["num_experts_per_tok"]

    @torch.no_grad()
    def _routed(self, x, idx, w):
        flat = idx.reshape(-1)
        y = torch.zeros_like(x)
        for e in flat.unique().tolist():
            slot = (flat == e).nonzero(as_tuple=True)[0]
            tok = slot // self.top_k
            out = self.experts[e](x[tok]) * w.reshape(-1, 1)[slot]
            y.index_add_(0, tok, out)
        return y

    def forward(self, x):
        shape = x.shape
        idx, w = self.gate(x)
        y = self._routed(x.view(-1, shape[-1]), idx, w.to(x.dtype)).view(shape)
        return y + self.shared_experts(x)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nh = cfg["num_attention_heads"]
        self.hd = cfg["hidden_size"] // self.nh
        self.theta = cfg["rope_theta"]
        h = cfg["hidden_size"]
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.nh, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.nh, self.hd).transpose(1, 2)
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, device=x.device).float() / self.hd))
        fr = torch.outer(torch.arange(S, device=x.device, dtype=inv.dtype), inv)
        emb = torch.cat((fr, fr), dim=-1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        dec = DECOMP["fns"]
        if dec is None:
            qs, ks = HOOKS["q"](q), HOOKS["k"](k)
            w = torch.matmul(qs, ks.transpose(2, 3)) / math.sqrt(self.hd)
        else:
            base, boost = dec
            q_b, q_B, k_b, k_B = base["q"](q), boost["q"](q), base["k"](k), boost["k"](k)
            w = (torch.matmul(q_b, k_B.transpose(2, 3)) + torch.matmul(q_B, k_b.transpose(2, 3))
                 - torch.matmul(q_b, k_b.transpose(2, 3))) / math.sqrt(self.hd)
        mask = torch.full((S, S), float("-inf"), device=x.device, dtype=w.dtype).triu(1)
        w = w + mask
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        if dec is None:
            out = torch.matmul(HOOKS["p"](w), HOOKS["v"](v))
        else:
            p_b, p_B, v_b, v_B = base["p"](w), boost["p"](w), base["v"](v), boost["v"](v)
            out = torch.matmul(p_b, v_B) + torch.matmul(p_B, v_b) - torch.matmul(p_b, v_b)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, -1))


class DecoderLayer(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        self.self_attn = Attention(cfg)
        moe = i >= cfg["first_k_dense_replace"] and i % cfg["moe_layer_freq"] == 0
        self.mlp = MoE(cfg) if moe else MLP(cfg["hidden_size"], cfg["intermediate_size"])
        self.input_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg["num_hidden_layers"]))
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])


class DeepseekMoEForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = Backbone(cfg)
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

    @property
    def device(self):
        return self.model.embed_tokens.weight.device

    @property
    def config(self):
        # lm_eval's HFLM (evaluation/zeroshot.py) expects a HF-style config object on any
        # pretrained model it wraps (self._model.config, tie_weights(), etc.) -- this port is a
        # plain nn.Module, not a transformers.PreTrainedModel, so both are supplied here. All
        # fields HFLM actually reads are getattr()-based with sane fallbacks (backend "causal",
        # max_length 2048) except model_type, which IS present in self.cfg (loaded from the real
        # checkpoint's config.json: "deepseek") and is set explicitly here for clarity anyway.
        return SimpleNamespace(**self.cfg)

    def tie_weights(self):
        pass  # checkpoint has tie_word_embeddings: False (config.json) -- lm_head is untied, no-op is correct

    def forward(self, input_ids, labels=None, use_cache=False, **_):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        logits = self.lm_head(self.model.norm(x))
        loss = None
        if labels is not None:
            # chunked fp32 CE: the full [S, 102400] fp32 logits + log-softmax copy cost
            # ~1.7GB, which OOM'd the last GPU (it also holds lm_head) at 22.5GB used
            lg = logits[:, :-1].reshape(-1, logits.shape[-1])
            lb = labels[:, 1:].to(lg.device).reshape(-1)
            n = (lb != -100).sum()
            tot = sum(F.cross_entropy(lg[i:i + 256].float(), lb[i:i + 256], ignore_index=-100, reduction="sum")
                      for i in range(0, lg.shape[0], 256))
            loss = tot / n
        return SimpleNamespace(loss=loss, logits=logits)  # not a fresh class per call: those are cyclic garbage that pins logits (0.4GB/step leak)


def load_config(path_or_dir=None):
    if path_or_dir is None:
        from huggingface_hub import hf_hub_download
        path_or_dir = hf_hub_download(MODEL_NAME, "config.json")
    if os.path.isdir(path_or_dir):
        path_or_dir = os.path.join(path_or_dir, "config.json")
    return json.load(open(path_or_dir))


def load_model(dtype=torch.bfloat16, device_map="auto"):
    """Meta-init, then accelerate shards the checkpoint across the visible GPUs."""
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from huggingface_hub import snapshot_download
    d = snapshot_download(MODEL_NAME, allow_patterns=["*.safetensors", "*.json"])
    cfg = load_config(d)
    with init_empty_weights():
        model = DeepseekMoEForCausalLM(cfg)
    # cap each GPU below its 24GB so activations (attention scores, logits) have headroom
    max_memory = {i: "17GiB" for i in range(torch.cuda.device_count())}
    # let overflow spill to host RAM instead of disk when few/no GPUs are visible; overridable for
    # boxes with less than 150GiB free (e.g. a shared server also running other jobs)
    max_memory["cpu"] = os.environ.get("DEEPSEEK_MAX_MEM_CPU", "150GiB")
    model = load_checkpoint_and_dispatch(model, d, device_map=device_map, dtype=dtype, max_memory=max_memory,
                                         no_split_module_classes=["DecoderLayer"])
    return model.eval()
