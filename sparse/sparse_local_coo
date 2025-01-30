from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
import os

from hellaswag import render_example, iterate_examples
from data import load_tokens, DataLoaderLite
from utils import get_most_likely_row

torch.set_printoptions(profile="full")

def get_sparse_attn_mask(n, attn_mode, local_attn_ctx=None, device='cpu'):
    """
    Builds a sparse attention mask using torch.sparse_coo_tensor() to avoid storing full [T, T] matrices.

    Args:
        n (int): Sequence length.
        attn_mode (str): Attention mode ('all', 'local', 'strided').
        local_attn_ctx (int, optional): Context size for 'local' and 'strided'.
        device (str or torch.device): Device to create tensors on.

    Returns:
        torch.sparse.FloatTensor: Sparse attention mask of shape [n, n].
    """
    indices = []
    values = []

    if attn_mode == 'all':
        # Full causal mask (lower triangular)
        for i in range(n):
            for j in range(i + 1):
                indices.append([i, j])
                values.append(1.0)

    elif attn_mode == 'local':
        # Local window: each token attends to up to `local_attn_ctx` past tokens
        bandwidth = local_attn_ctx
        for i in range(n):
            start = max(0, i - bandwidth)
            for j in range(start, i + 1):
                indices.append([i, j])
                values.append(1.0)

    elif attn_mode == 'strided':
        # Strided attention: each token attends to every `stride`th past token
        stride = local_attn_ctx
        for i in range(n):
            for j in range(0, i + 1, stride):
                indices.append([i, j])
                values.append(1.0)

    else:
        raise ValueError(f"Unknown attn_mode: {attn_mode}")

    if len(indices) == 0:
        # Avoid creating an empty sparse tensor
        indices = torch.empty((2, 0), dtype=torch.long, device=device)
        values = torch.empty((0,), dtype=torch.float, device=device)
    else:
        indices = torch.tensor(indices, dtype=torch.long, device=device).T      # Shape [2, nnz]
        values = torch.tensor(values, dtype=torch.float, device=device)         # Shape [nnz]

    sparse_mask = torch.sparse_coo_tensor(indices, values, (n, n), device=device)
    return sparse_mask


def blocksparse_attention_impl(q, k, v, attn_mode, local_attn_ctx=None, blocksize=32, debug_print=True):
    """
    Implements block-sparse attention using sparse attention masks.
    This function avoids computing attention scores for masked positions.

    Args:
        q (torch.Tensor): Query tensor of shape [B, H, T, D].
        k (torch.Tensor): Key tensor of shape [B, H, T, D].
        v (torch.Tensor): Value tensor of shape [B, H, T, D].
        attn_mode (str): Attention mode ('all', 'local', 'strided').
        local_attn_ctx (int, optional): Context size for 'local' and 'strided'.
        blocksize (int): Block size for 'strided' attention.
        debug_print (bool): If True, prints debug information.

    Returns:
        torch.Tensor: Attention output of shape [B, H, T, D].
    """
    device = q.device
    B, H, T, D = q.shape  # Batch, Heads, Time, Head_dim
    scale_amount = 1.0 / math.sqrt(D)

    # Compute raw attention scores
    # [B, H, T, D] x [B, H, D, T] -> [B, H, T, T]
    # To leverage sparsity, we compute only the required elements
    # However, PyTorch does not support sparse matmul directly, so we proceed cautiously
    # First, compute the dense attention scores
    # TODO: Implement efficient sparse matmul if possible
    # For now, proceed with dense matmul and masking

    # Note: This implementation still uses dense matmul due to PyTorch's limitations
    # To truly leverage sparsity, custom CUDA kernels or specialized libraries are needed

    # Compute raw attention scores: [B, H, T, D] x [B, H, D, T] -> [B, H, T, T]
    # Can use torch.einsum for clarity. Alternatively, torch.matmul(q, k.transpose(-2, -1)).
    # To potentially save memory, use torch.bmm for each head, but since we're working with batched heads, proceed with matmul
    scores = torch.matmul(q, k.transpose(-2, -1))  # [B, H, T, T]
    scores = scores * scale_amount

    # Get sparse attention mask
    sparse_mask = get_sparse_attn_mask(T, attn_mode, local_attn_ctx=local_attn_ctx, device=device)  # [T, T]

    # Convert sparse_mask to dense for masking since softmax requires dense tensors.
    # Alternatively, implement a masked softmax that skips masked positions.
    # However, PyTorch does not support masked softmax for sparse tensors, so we need to convert to dense.
    mask_dense = sparse_mask.to_dense()  # [T, T]

    # Expand mask to [1, 1, T, T] for broadcasting
    mask_dense = mask_dense.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

    # Apply mask: set masked positions to -inf
    scores = scores.masked_fill(mask_dense == 0, float('-inf'))  # [B, H, T, T]

    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)  # [B, H, T, T]

    # Apply attention weights to values
    attn_output = torch.matmul(attn_weights, v)  # [B, H, T, D]

    # Optional debug prints
    if debug_print:
        nnz = sparse_mask._nnz()
        total = T * T
        sparsity = nnz / total
        print(f"[DEBUG] blocksparse_attention_impl: attn_mode={attn_mode}")
        print(f"[DEBUG] Mask Sparsity: {sparsity:.4f} ({nnz}/{total})")
        if T <= 64:
            print(f"[DEBUG] Mask (first {T} tokens):\n{mask_dense[0, 0, :T, :T]}")
        else:
            print(f"[DEBUG] Mask sample (0:8,0:8):\n{mask_dense[0, 0, 0:8, 0:8]}")

    return attn_output


class CausalSelfAttention(nn.Module):
    """
    Implements causal self-attention with optional sparsity.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding size must be divisible by number of heads."

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.attn_mode = config.attn_mode               # 'all', 'local', 'strided'
        self.local_attn_ctx = config.local_attn_ctx
        self.blocksize = config.blocksize

        # Query, Key, Value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # Custom scaling attribute

        # Buffer for legacy purposes (not used in sparse attention)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
            persistent=False
        )

    def forward(self, x):
        """
        Forward pass for causal self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, C].

        Returns:
            torch.Tensor: Output tensor of shape [B, T, C].
        """
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape and transpose to [B, H, T, D]
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Apply sparse attention
        attn_output = blocksparse_attention_impl(
            q, k, v,
            attn_mode=self.attn_mode,
            local_attn_ctx=self.local_attn_ctx,
            blocksize=self.blocksize,
            debug_print=True                                # Set to True for debugging
        )

        # Reshape back to [B, T, C]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # Final projection
        y = self.c_proj(attn_output)

        return y


class MLP(nn.Module):
    """
    Implements the MLP component of a Transformer block.
    """

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
    """
    Implements a single Transformer block consisting of
    LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-LN, then Attention, then Residual
        x = x + self.attn(self.ln_1(x))
        # Pre-LN, then MLP, then Residual
        x = x + self.mlp(self.ln_2(x))
        return x



@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    attn_mode: str = 'all'          # 'all', 'local', 'strided'
    local_attn_ctx: int = 32        # Context size for 'local' and 'strided'
    blocksize: int = 32             # Block size for 'strided'



class GPT(nn.Module):
    """
    Implements the GPT model with block-sparse attention.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Transformer backbone
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),                  # Token embeddings
            'wpe': nn.Embedding(config.block_size, config.n_embd),                  # Positional embeddings
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),     # Transformer blocks
            'ln_f': nn.LayerNorm(config.n_embd),  # Final LayerNorm
        })

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)      # Output projection

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initializes weights following the specified scheme.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            # Adjust std if NANOGPT_SCALE_INIT is set
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass of the GPT model.

        Args:
            idx (torch.Tensor): Input token indices of shape [B, T].
            targets (torch.Tensor, optional): Target token indices of shape [B, T].

        Returns:
            tuple: (logits, loss)
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}."

        # Token and positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)     # [B, T, C]
        pos_emb = self.transformer.wpe(pos)     # [T, C]
        x = tok_emb + pos_emb                   # [B, T, C]

        # Pass through Transformer blocks
        for block in self.transformer.h:
            x = block(x)                        # [B, T, C]

        # Final LayerNorm
        x = self.transformer.ln_f(x)            # [B, T, C]

        # Language Modeling Head
        logits = self.lm_head(x)                # [B, T, vocab_size]
        loss = None

        if targets is not None:
            # Compute cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            print(loss)

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str):
        """
        Loads pretrained GPT-2 model weights from Hugging Face and maps them to this GPT model.

        Args:
            model_type (str): One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'.

        Returns:
            GPT: GPT model with loaded weights.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}, "Unsupported model type."
        from transformers import GPT2LMHeadModel

        print(f"Loading weights from pretrained GPT-2 model: {model_type}")

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        config_args.update({
            'vocab_size': 50257,
            'block_size': 1024,
            'attn_mode': 'all',  # Default to 'all' for pretrained models
            'local_attn_ctx': 32,
            'blocksize': 32,
        })

        # Create GPTConfig and GPT model
        config = GPTConfig(**config_args)
        model = cls(config)

        # Load Hugging Face GPT-2 weights
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Get the state dictionaries, excluding certain keys
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]

        # Define which weights need to be transposed
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Ensure matching key lengths
        assert len(sd_keys_hf) == len(sd_keys), f"Mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        # Map Hugging Face weights to this model
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Transpose weights where necessary
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"Shape mismatch for {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Directly copy weights
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch for {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        """
        Configures the optimizer with separate parameter groups for weight decay.

        Args:
            weight_decay (float): Weight decay factor.
            learning_rate (float): Learning rate.
            device_type (str): 'cuda' or 'cpu'.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        # Separate parameters for weight decay and no weight decay
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"Number of decayed parameter tensors: {len(decay_params)}, with {num_decay:,} parameters")
        print(f"Number of non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay:,} parameters")

        import inspect
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = (fused_available and device_type == "cuda")
        print(f"Using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )
        return optimizer
