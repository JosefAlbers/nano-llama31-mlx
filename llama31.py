import os
import glob
import fire
import time
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict

from tokenizer import Tokenizer

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download

# -----------------------------------------------------------------------------
# ModelArgs

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False
    max_batch_size: int = 32
    max_seq_len: int = 2048
    flash: bool = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

# -----------------------------------------------------------------------------
# Transformer

def apply_scaling(freqs):
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return mx.array(new_freqs, dtype=freqs.dtype)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2)[:dim // 2].astype(mx.float32) / dim))
    t = mx.arange(end, dtype=mx.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = mx.outer(t, freqs)
    freqs_cis = np.exp(1j * freqs)
    freqs_cis_real = mx.stack([mx.array(freqs_cis.real), mx.array(freqs_cis.imag)], axis=-1)
    return np.array(freqs_cis_real)

def apply_rotary_emb(x, freqs_cis):
    xshaped = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.reshape(1, xshaped.shape[1], 1, xshaped.shape[3], 2)
    x_out2_real = xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1]
    x_out2_imag = xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1]
    x_out2 = mx.stack([x_out2_real, x_out2_imag], axis=-1)
    x_out2 = x_out2.reshape(*x_out2.shape[:-2], -1)
    return x_out2.astype(x.dtype)

def repeat_kv(x, n_rep: int):
    if n_rep == 1:
        return x
    bs, slen, n_kv_heads, head_dim = x.shape
    x = x[:,:,:,None,:]
    x = mx.repeat(x, repeats=n_rep, axis=3).reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    return x

class KVCache:
    def __init__(self, batch_size, seq_length, n_kv_heads, head_dim, dtype):
        cache_shape = (batch_size, seq_length, n_kv_heads, head_dim)
        self.cache_k = mx.zeros(cache_shape, dtype=dtype)
        self.cache_v = mx.zeros(cache_shape, dtype=dtype)

    def update(self, start_pos, xk, xv):
        seqlen = xk.shape[1]
        self.cache_k[:, start_pos : start_pos + seqlen] = xk
        self.cache_v[:, start_pos : start_pos + seqlen] = xv
        xk = self.cache_k[:, : start_pos + seqlen]
        xv = self.cache_v[:, : start_pos + seqlen]
        return xk, xv

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.flash = args.flash
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False )
        self.k_proj = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.cache = None

    def __call__(
        self,
        x,
        start_pos,
        freqs_cis,
        mask,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.reshape(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        if self.cache is not None:
            xk, xv = self.cache.update(start_pos, xk, xv)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)
        xq, xk, xv = (x.transpose(0, 2, 1, 3) for x in (xq, xk, xv))
        # if self.flash:
        #     output = mx.fast.scaled_dot_product_attention(xq, xk, xv, scale=self.scale, mask=mask)
        # else:
        if True:
            scores = (xq * self.scale) @ xk.transpose(0, 1, 3, 2)
            if mask is not None:
                scores = scores + mask
            scores = mx.softmax(scores, axis=-1)
            output = scores @ xv
        output = output.transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)
        proj = self.o_proj(output)

        return proj

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.self_attn = Attention(args)
        self.mlp = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.input_layernorm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(
        self,
        x,
        start_pos: int,
        freqs_cis,
        mask,
    ):
        h = x + self.self_attn(self.input_layernorm(x), start_pos, freqs_cis, mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.embed_tokens = nn.Embedding(params.vocab_size, params.dim)
        self.layers = [TransformerBlock(params) for _ in range(params.n_layers)]
        self.norm = nn.RMSNorm(params.dim, eps=params.norm_eps)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )

    def __call__(self, tokens, start_pos):
        _bsz, seqlen = tokens.shape
        h = self.embed_tokens(tokens)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = mx.triu(mx.full((seqlen, seqlen), -mx.inf), k=1)
            mask = np.concatenate(
                [np.zeros((seqlen, start_pos)), mask],
                axis=1
            )

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        return h

class Model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.model = Transformer(params)
        self.lm_head = nn.Linear(params.dim, params.vocab_size, bias=False)

    def __call__(self, tokens, start_pos):
        h = self.model(tokens, start_pos)
        h = self.lm_head(h)
        return h

# -----------------------------------------------------------------------------
# Llama wrapper

class Llama:

    @staticmethod
    def build(
        repo_id: str,
        max_seq_len: int,
        max_batch_size: int,
        flash: bool = False,
    ) -> "Llama":

        with open("Meta-Llama-3.1-8B-Instruct/params.json", "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            flash=flash,
            **params,
        )
        model = Model(model_args)
        _ckpt_dir = snapshot_download("meta-llama/Meta-Llama-3.1-8B-Instruct", allow_patterns=["*.safetensors"], token=os.getenv('HF_JOSEF_TOKEN'))
        model_wt = [(k, v) for wf in glob.glob(f"{_ckpt_dir}/*.safetensors") for k, v in mx.load(wf).items()]
        model.load_weights(model_wt)
        mx.eval(model.parameters())
        model.eval()
        tokenizer = Tokenizer("Meta-Llama-3.1-8B-Instruct/tokenizer.model")
        return Llama(model, tokenizer)

    def __init__(self, model, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        params = self.model.params
        bsz = len(prompt_tokens)
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)
        for block in self.model.model.layers:
            layer_dtype = block.self_attn.q_proj.weight.dtype
            block.self_attn.cache = KVCache(
                batch_size=bsz,
                seq_length=total_len,
                n_kv_heads=params.n_kv_heads,
                head_dim=params.dim // params.n_heads,
                dtype=layer_dtype,
            )
        pad_id = self.tokenizer.pad_id
        tokens = mx.full((bsz, total_len), pad_id)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = mx.array(t)
        prev_pos = 0
        eos_reached = mx.array([False] * bsz)
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model(tokens, prev_pos)
        stop_tokens = mx.array(list(self.tokenizer.stop_tokens))
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model(tokens[:, prev_pos:cur_pos], prev_pos)
            next_token = mx.argmax(logits[:, -1], axis=-1)
            next_token = next_token.reshape(-1)
            next_token = mx.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # eos_reached |= (~input_text_mask[:, cur_pos]) & (
            #     torch.isin(next_token, stop_tokens)
            # )
            prev_pos = cur_pos
            if all(eos_reached):
                break
        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                except ValueError:
                    pass
            out_tokens.append(toks)
        for block in self.model.model.layers:
            block.self_attn.cache = None
        return out_tokens

    def text_completion(
        self,
        prompts: List[str],
        max_gen_len: Optional[int] = None,
        echo: bool = False,
    ):
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            echo=echo,
        )
        completions = [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]
        return completions

# -----------------------------------------------------------------------------
# int main

def main(
    repo_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_seq_len: int = 256,
    max_gen_len: int = 256,
    max_batch_size: int = 8,
    flash: bool = True,
):
    llama = Llama.build(
        repo_id=repo_id,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        flash=flash,
    )
    prompts: List[str] = [
        "Once upon a time",
        "One day",
        "Lily and George were best friends",
        "On a dark and stormy night",
    ]
    t0 = time.time()
    results = llama.text_completion(
        prompts,
        max_gen_len=max_gen_len,
    )
    t1 = time.time()
    print(f"Generated in {t1 - t0:.2f} seconds")
    for prompt, result in zip(prompts, results):
        print(prompt, end="")
        print(f"{result['generation']}")
        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)
