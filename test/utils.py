from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from hellaswag import render_example, iterate_examples
import tiktoken
import numpy as np
import os

def get_most_likely_row(tokens, mask, logits):

    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()

    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)

    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)

    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask

    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)

    pred_norm = avg_loss.argmin().item()

    return pred_norm
