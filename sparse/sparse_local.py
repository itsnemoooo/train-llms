from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
import numpy as np
import tiktoken

from hellaswag import render_example, iterate_examples
from data import load_tokens, DataLoaderLite
from utils import get_most_likely_row

"""
# basic sparse transformer implementation with a local pattern of 32 tokens.
# replaces the following classes in the baseline model: CausalSelfAttention, MLP, Block, GPTConfig, GPT.
# might have to make arrangements for precision.
# downside: NO FLASH ATTENTION, open AI implementation may have it. 
# https://github.com/kyegomez/SparseAttention
"""


torch.set_printoptions(profile="full")


def get_attn_mask(n, attn_mode, local_attn_ctx=None, device='cpu'):
    """
    Builds a causal attention mask of shape [1, 1, n, n].
    'attn_mode' can be:
      - 'all': full causal (token i sees [0...i])
      - 'local': a local band of size `local_attn_ctx`
      - 'strided': tokens i see those j for which (i-j) mod stride == 0 and j <= i
    """
    if attn_mode == 'all':
        # Full causal mask
        b = torch.tril(torch.ones([n, n], device=device))
    elif attn_mode == 'local':
        # Local window = local_attn_ctx
        bandwidth = local_attn_ctx
        ctx = min(n, bandwidth)
        full_tril = torch.tril(torch.ones([n, n], device=device))
        shifted_tril = torch.tril(torch.ones([n, n], device=device), diagonal=-ctx)
        b = full_tril - shifted_tril
        # This yields a band of width `ctx+1` around the diagonal, ensuring token i sees up to ctx tokens back.
    elif attn_mode == 'strided':
        stride = local_attn_ctx
        x = torch.reshape(torch.arange(n, dtype=torch.int32, device=device), [n, 1])
        y = x.transpose(0, 1)
        z = torch.zeros([n, n], dtype=torch.int32, device=device)
        q = z + x  # shape [n, n], q[i,j] = i
        k = z + y  # shape [n, n], k[i,j] = j
        c1 = q >= k
        c2 = ((q - k) % stride) == 0
        c3 = torch.logical_and(c1, c2)
        b = c3.float()
    else:
        raise ValueError(f"Unknown attn_mode: {attn_mode}")

    b = b.view(1, 1, n, n)  # broadcast over [batch, heads, T, T]
    return b

def strided_transpose(x, n_ctx, local_attn_ctx, blocksize):
    """
    Re-arranges the (batch, T, head_dim) data for 'strided' attention.
    We assert (n_ctx // local_attn_ctx) is divisible by blocksize.
    """
    bT_ctx = n_ctx // local_attn_ctx
    assert bT_ctx % blocksize == 0, f"bT_ctx={bT_ctx}, blocksize={blocksize} mismatch."
    n, t, embd = x.size()
    # reshape [n, t, embd] -> [n, bT_ctx, local_attn_ctx, embd]
    x = x.reshape(n, bT_ctx, local_attn_ctx, embd)
    # transpose -> [n, local_attn_ctx, bT_ctx, embd]
    x = x.transpose(1, 2)
    # flatten back
    x = x.reshape(n, t, embd)
    return x

def blocksparse_attention_impl(q, k, v, attn_mode, local_attn_ctx=None, blocksize=32, debug_print=True):
    """
    Manually implements scaled dot-product attention with a custom mask for 'all', 'local', or 'strided'.
    q, k, v shape: [B, heads, T, head_dim]
    Returns: [B, heads, T, head_dim]
    """
    device = q.device
    B, H, T, D = q.shape  # batch, heads, time, head_dim
    scale_amount = 1.0 / math.sqrt(D)

    # If 'strided', first re-arrange the data
    if attn_mode == 'strided':
        q = strided_transpose(q, T, local_attn_ctx, blocksize)
        k = strided_transpose(k, T, local_attn_ctx, blocksize)
        v = strided_transpose(v, T, local_attn_ctx, blocksize)

    # Compute raw scores
    # [B, heads, T, head_dim] x [B, heads, T, head_dim]^T -> [B, heads, T, T]
    w = torch.matmul(q, k.transpose(-2, -1))
    w = w * scale_amount

    # Build mask [1, 1, T, T] for the chosen attn_mode
    mask = get_attn_mask(T, attn_mode, local_attn_ctx=local_attn_ctx, device=device)
    # w shape is [B, H, T, T], mask shape is [1, 1, T, T]
    w = w.masked_fill(mask == 0, float('-inf'))

    # Softmax along last dimension
    w = F.softmax(w, dim=-1)

    # Weighted sum
    a = torch.matmul(w, v)  # shape [B, H, T, D]

    # If 'strided', undo the prior reshape
    if attn_mode == 'strided':
        # revert the strided transpose
        n, h, t, embd = a.size()
        bT_ctx = T // local_attn_ctx
        a = a.reshape(n, h, local_attn_ctx, bT_ctx, embd)       # [B, H, local_attn_ctx, bT_ctx, D]
        a = a.transpose(2, 3)                       # [B, H, bT_ctx, local_attn_ctx, D]
        a = a.reshape(n, h, t, embd)                            # back to [B, H, T, D]

    # Optional debug prints
    if debug_print:
        print(f"[DEBUG] blocksparse_attention_impl: attn_mode={attn_mode}")
        total_els = mask.numel()
        nonzero_els = mask.sum().item()
        frac = nonzero_els / total_els
        print(f"[DEBUG] Mask Sparsity: {frac:.4f} ({nonzero_els}/{total_els})")
        if T <= 64:
            print(f"[DEBUG] Mask (first {T} tokens):\n{mask[0,0,:T,:T]}")
        else:
            print(f"[DEBUG] Mask sample (0:8,0:8):\n{mask[0,0,0:8,0:8]}")

    return a


class CausalSelfAttention(nn.Module):
    """
    A drop-in replacement for the 'standard' CausalSelfAttention from snippet #2.
    The difference: We can do 'all', 'local', or 'strided' attention using
    blocksparse_attention_impl, or revert to PyTorch 2.0's built-in if attn_mode='all'.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.attn_mode = getattr(config, 'attn_mode', 'all')
        self.local_attn_ctx = getattr(config, 'local_attn_ctx', 32)
        self.blocksize = getattr(config, 'blocksize', 32)

        # Query, Key, Value projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # This buffer used to hold a causal mask in older code.
        # We'll keep it here to "respect the nuance" but we won't rely on it
        # unless we do 'all' mode using the manual approach.
        self.register_buffer("bias",
                             torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1,config.block_size,config.block_size),
                             persistent=False)

    def forward(self, x):
        B, T, C = x.size()
        # Project to Q,K,V
        qkv = self.c_attn(x)  # [B, T, 3*C]
        q, k, v = qkv.split(self.n_embd, dim=2)

        # [B, T, n_embd] -> [B, n_head, T, head_dim]
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        if self.attn_mode == 'all':

            # Option A: rely on PyTorch 2.0 built-in function with is_causal=True
            # y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            # Option B: do the same “manual” approach to match local/strided
            y = blocksparse_attention_impl(
                q, k, v,
                attn_mode='all',
                local_attn_ctx=self.local_attn_ctx,
                blocksize=self.blocksize,
                debug_print=False
            )
        else:
            # local or strided
            y = blocksparse_attention_impl(
                q, k, v,
                attn_mode=self.attn_mode,
                local_attn_ctx=self.local_attn_ctx,
                blocksize=self.blocksize,
                debug_print=True
            )

        # [B, n_head, T, head_dim] -> [B, T, n_embd]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Final projection
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-LN, then attn, + residual
        x = x + self.attn(self.ln_1(x))
        # Pre-LN, then MLP, + residual
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    attn_mode: str = 'all'      # can be 'all', 'local', or 'strided'
    local_attn_ctx: int = 32
    blocksize: int = 32

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Transformer "backbone"
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        # Language Model Head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Matches the second snippet's logic of setting an adjusted std
        if NANOGPT_SCALE_INIT is found, else default 0.02.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            # scale down if we find "NANOGPT_SCALE_INIT"
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx: (B, T)
        targets: (B, T) or None
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} > block size {self.config.block_size}"

        # Token + Position Embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)  # [B, T, n_embd]
        pos_emb = self.transformer.wpe(pos)  # [T, n_embd]
        x = tok_emb + pos_emb

        # Pass through the layers
        for block in self.transformer.h:
            x = block(x)

        # Final LayerNorm
        x = self.transformer.ln_f(x)

        # LM Head
        logits = self.lm_head(x)  # [B, T, vocab_size]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            print(loss)

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str):
        """
        Exactly like snippet #2, loads GPT-2 from HF and remaps weights.
        We'll just replicate the logic, including the calls for transposed weights.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")

        config_args = {
            'gpt2':        dict(n_layer=12,  n_head=12,  n_embd=768),
            'gpt2-medium': dict(n_layer=24,  n_head=16,  n_embd=1024),
            'gpt2-large':  dict(n_layer=36,  n_head=20,  n_embd=1280),
            'gpt2-xl':     dict(n_layer=48,  n_head=25,  n_embd=1600),
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['attn_mode'] = 'all'
        config_args['local_attn_ctx'] = 32
        config_args['blocksize'] = 32

        # Create our config
        config = GPTConfig(**config_args)
        # Create our GPT
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        # Load HF GPT-2
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Check matching lengths
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"Shape mismatch on {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch on {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        """
        Identical to snippet #2: separate param groups for weight decay or not,
        then optionally use fused AdamW if device_type == "cuda" and it's available.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # decay for 2D params, no decay for biases / LN / etc.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"num decayed param tensors: {len(decay_params)}, {num_decay} params total")
        print(f"num no-decay param tensors: {len(nodecay_params)}, {num_nodecay} params total")

        import inspect
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = (fused_available and device_type == "cuda")
        print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )
        return optimizer

