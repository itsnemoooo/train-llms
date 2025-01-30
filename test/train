from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
import numpy as np
import tiktoken

from model import CausalSelfAttention, MLP, Block, GPTConfig, GPT
from data import load_tokens, DataLoaderLite
from utils import get_most_likely_row
from hellaswag import render_example, iterate_examples

def main():
    import sys
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    ddp = False

    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA."
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        device_type = "cuda"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        print(f"Using device: {device}")
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        device_type = "cpu"
        print("---------------------------------------------")
        print(f"Using device: {device}")
        print(f"Using device_type: {device_type}")

    torch.manual_seed(1337)

    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")

    total_batch_size = 1024
    B = 8
    T = 128

    assert total_batch_size % (B * T * ddp_world_size) == 0, "Ensure total_batch_size is divisible by B * T * ddp_world_size"

    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

    if master_process:
        print("---------------------------------------------")
        print(f"Total desired batch size: {total_batch_size}")
        print(f"Calculated gradient accumulation steps: {grad_accum_steps}")
        print("---------------------------------------------")


    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")
    print("---------------------------------------------")

    torch.set_float32_matmul_precision('high')

    model = GPT(GPTConfig(vocab_size=50304)).to(device)

    # Disable torch.compile for testing
    use_compile = False
    if use_compile:
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model

    torch.set_float32_matmul_precision("high")

    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

    # Learning rate schedule parameters adjusted for testing
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    max_steps = 10
    warmup_steps = 2

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps

        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        decay_ratio = max(0.0, min(decay_ratio, 1.0))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

        return min_lr + coeff * (max_lr - min_lr)


    if device_type == 'cpu':
        autocast_dtype = torch.float32  #float32, bfloat16
        print("---------------------------------------------")
        print("Using torch.float32 for CPU autocast")
        print("---------------------------------------------")


    for step in range(max_steps):
        last_step = (step == max_steps - 1)

        # Validation
        if step % 1 == 0 or last_step:
            raw_model.eval()
            val_loader.current_shard = 0
            val_loader.tokens = load_tokens(val_loader.shards[val_loader.current_shard])
            val_loader.current_position = val_loader.B * val_loader.T * val_loader.process_rank

            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 2
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                        logits, loss = raw_model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

            if master_process:
                print(f"Validation loss: {val_loss_accum.item():.4f}")

        # HellaSwag Evaluation
        if step % 1 == 0 or last_step:
            num_correct_norm = 0
            num_total = 0

            for i, example in enumerate(iterate_examples("val")):
                if i >= 10:
                    break
                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                        logits, _ = raw_model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)

                num_total += 1
                num_correct_norm += int(pred_norm == label)

            if ddp:
                num_total_tensor = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm_tensor = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm_tensor, op=dist.ReduceOp.SUM)
                num_total = num_total_tensor.item()
                num_correct_norm = num_correct_norm_tensor.item()

            acc_norm = num_correct_norm / num_total if num_total > 0 else 0.0

            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total} = {acc_norm:.4f}")


        # Generation
        if (step > 0 and step % 1 == 0) or last_step:
            raw_model.eval()
            num_return_sequences = 1
            max_length = 16
            tokens_input = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens_input, dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1).to(device)
            xgen = tokens.clone()
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 * ddp_rank)

            while xgen.size(1) < max_length:
                with torch.no_grad():
                    if torch.any(torch.isnan(xgen)) or torch.any(torch.isinf(xgen)):
                        raise ValueError("Input tensor contains NaN or Inf values.")

                    logits, _ = raw_model(xgen)
                    logits = logits[:, -1, :]
                    if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                        raise ValueError("Logits contain NaN or Inf values.")

                    probs = F.softmax(logits, dim=-1)
                    if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)):
                        raise ValueError("Probabilities contain NaN or Inf values.")

                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    if torch.any(topk_probs < 0):
                        raise ValueError("Top-k probabilities contain negative values.")

                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix)
                    xgen = torch.cat((xgen, xcol), dim=1)
                    # print("Generated a token without crashing.")

            for i in range(num_return_sequences):
                tokens_generated = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens_generated)
                print(f"Rank {ddp_rank} Sample {i}: {decoded}")

        # Training Loop
        raw_model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                logits, loss = raw_model(x, y)

            loss = loss / grad_accum_steps
            loss_accum += loss.detach()

            if ddp:
                raw_model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)


        # Gradient Clipping
        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        print(f"Gradient norm: {norm:.4f}")

        # Update Learning Rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print("Optimizer step completed.")

        optimizer.step()

        if master_process:
            print(f"Step: {step}, Loss: {loss_accum.item():.6f}, LR: {lr:.4e}, Norm: {norm:.4f}")
            print("--------------------------")


if __name__ == "__main__":
    main()
