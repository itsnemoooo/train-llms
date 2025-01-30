import pytest
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig, CausalSelfAttention, MLP, Block

@pytest.fixture
def default_config():
    return GPTConfig()

@pytest.fixture
def modified_config():
    return GPTConfig(n_layer=6, n_head=8, n_embd=512)

def test_causal_self_attention_forward(default_config):
    attn = CausalSelfAttention(default_config)
    x = torch.randn(2, 1024, default_config.n_embd)
    y = attn(x)
    assert y.shape == x.shape, "CausalSelfAttention output shape mismatch"

def test_mlp_forward(default_config):
    mlp = MLP(default_config)
    x = torch.randn(2, 1024, default_config.n_embd)
    y = mlp(x)
    assert y.shape == x.shape, "MLP output shape mismatch"

def test_block_forward(default_config):
    block = Block(default_config)
    x = torch.randn(2, 1024, default_config.n_embd)
    y = block(x)
    assert y.shape == x.shape, "Block output shape mismatch"

def test_gpt_forward(default_config):
    model = GPT(default_config)
    idx = torch.randint(0, default_config.vocab_size, (2, 1024))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 1024, default_config.vocab_size), "GPT logits shape mismatch"
    assert loss is not None, "GPT loss should not be None when targets are provided"

def test_gpt_forward_no_targets(default_config):
    model = GPT(default_config)
    idx = torch.randint(0, default_config.vocab_size, (2, 1024))
    logits, loss = model(idx)
    assert logits.shape == (2, 1024, default_config.vocab_size), "GPT logits shape mismatch"
    assert loss is None, "GPT loss should be None when targets are not provided"

def test_gpt_with_modified_config(modified_config):
    model = GPT(modified_config)
    idx = torch.randint(0, modified_config.vocab_size, (2, 1024))
    logits, loss = model(idx)
    assert logits.shape == (2, 1024, modified_config.vocab_size), "GPT logits shape mismatch with modified config"


