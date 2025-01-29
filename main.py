from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from hellaswag import render_example, iterate_examples



#-------------------------------------------------------
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT= 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        #self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        # blocks size == sequence length 
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att - att.max(dim=-1, keepdim=True)[0], dim=-1)
        # y = att @ v #weighted sum of interesting tokens at every token
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        #flash attention is memory-aware but does not torch.compile. they avoid the load-store into HBM


        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__ (self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU() # replaced from original gpt-2 paper (used tanh approx)
        self.c_proj = nn.Linear( 4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT= 1

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
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass #decorator
class GPTConfig:
    block_size: int  = 1024
    vocab_size: int = 50257 #ugly number. increase to 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module): #construct gpt class

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict( #elements of transofmer in non-sequential way
            wte = nn.Embedding(config.vocab_size, config.n_embd), #token embeddings / output embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), #positional encoding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #hidden. 'n' layer blocks. everything containd in the grey block
            ln_f = nn.LayerNorm(config.n_embd), #final layer norm the linear part outside the transformer 
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # final projection
        #similar tokens should have similar embeddings
        #re-use wte, matmul of output of transformer and wte. we want gradient contributions from both branches
        #weight sharing scheme "Why should I learn two separate representations for a word? The embeddings I learned for input should be good enough for output predictions too."
        self.transformer.wte.weight = self.lm_head.weight 
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer)**-0.5 #2 blocks adding to residual pathway for the Linear Layer.
                #Scaling down std deviation based on number of layers. the 2 comes from the the attention + mlp
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02) #with javier it would be 1/sqrt(#features)... 1/sqrt(768) for ex
            if module.bias is not None: #initialise bias to 0, NOT DEFAULT for PyTorch
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


#nn embedding is a wrapper module around a single block of numbers (tensor) that allows us to access elemtns by indexing into the rows
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)... batch dimension and Time dimension
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T).. like range but for pytorch
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None: #flatten tensor and targets to calculate loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    @classmethod
    def from_pretrained(cls, model_type): #class method that returns GPT object if we give it model type
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params. 25 is not a nice number.
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict() #create the state dict for our model 
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict() #create the state dict for the hf model 

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] #1d tensors, no need to weight decay
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        import inspect
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        #fused doesn't launch lots of kernels for for loop. fuse them into single kernel and call an single update kernel for adamW
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


    
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

#-------------------------------------------------------
import os
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train','val'}

        #load shards
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        #state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self): #iterate all the shards
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs. the dimensions in the .view are B and T
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard+1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 64 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# get logits
torch.set_float32_matmul_precision("high") #tells pytorch to run kernels in float 32. TF32 used when available. Every place with nn.Linear the matmul is running on tensorcores 
#model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig(vocab_size=50304)).to(device) # Ensure the model is moved to device

model = torch.compile(model) # analyses your model. operations you want to use. doesn't run in eager mode. able to understand what operations you want to run.

#removes python interpreter from forward pass entirely. compiles neural network as an object.
#we make a bunch of trips from HBM to GPU without using torch.compile. we can speed up with torch.compile using Kernel fusion. it knows the entire functions so it schedules read/write from memory.
# ther is memory in the chip but not a lot of memory.

#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps= 1e-8) #using values from gpt3 paper

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate = 6e-4, device_type=device_type)
#returns optimizer objecte

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
    pass
#learning rate with cosine decay 

max_lr = 6e-4
min_lr = max_lr * 0.1
max_steps = 100000
warmup_steps = 715

def get_lr(it):
    #1) Warmup region
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps #'it + 1' is so we don't start w/ learning rate of 0
    #2) if it > lr_decay_iters, return min learning rate
    if it > warmup_steps:
        return min_lr
    #3) in between, we use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

import time

# resume_checkpoint = "./log/model_15000.pt"
# if os.path.exists(resume_checkpoint):
#     if master_process:
#         print(f"Loading checkpoint from {resume_checkpoint}")
#         ckpt = torch.load(resume_checkpoint, map_location=device)
#         raw_model.load_state_dict(ckpt['model'])
#         optimizer.load_state_dict(ckpt['optimizer'])
#         start_step = ckpt['step'] + 1

#         # Make sure RNG state is a CPU ByteTensor
#         cpu_rng_state = ckpt['rng_state']
#         if cpu_rng_state.device.type != 'cpu':
#             cpu_rng_state = cpu_rng_state.cpu()
#         torch.set_rng_state(cpu_rng_state)

#         # For CUDA RNG
#         cuda_rng_state = ckpt['cuda_rng_state']
#         if cuda_rng_state.device.type != 'cpu':
#             cuda_rng_state = cuda_rng_state.cpu()
#         torch.cuda.set_rng_state(cuda_rng_state, device=device)
#     else:
#         start_step = 0
    
#     # If DDP, broadcast parameters/optimizer from rank 0...
# else:
#     print("No checkpoint found. Starting from scratch.")
#     start_step = 0


start_time = time.time()
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
# validation split
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device),y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x,y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}") 
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.6f}\n")
    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        #enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 * ddp_rank)

        while xgen.size(1) < max_length:
            
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)

        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i} {decoded}")

    #evaluate on hellaswag

    #training loop
    model.train()        
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x,y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16): #forward pass block in fp16. activations tesnor is bf16. transformer.wte.dtype is still fp32. activations changed to 16.
            logits, loss = model(x,y)
        
        loss = loss / grad_accum_steps #because of cross-entropy is also mean-loss reduciton. we are summing the losses max_steps times. we need to normalise the loss.
        loss_accum += loss.detach() # for printing
        loss.backward() #contains the += operation for accum gradients
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
        if step > 0 and (step % 5000 == 0 or last_step):
            checkpoint_path = os.path.join(log_dir, f'model_{step:05d}.pt')
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'config': raw_model.config,
                'step': step,
                'val_loss': val_loss_accum.item(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(device=device),
            }
            torch.save(checkpoint, checkpoint_path)
        #max norm through gradient clipping. prevents the model from getting gradient shocked. exploding gradients.
        #model starts off quite random and unstable.

    norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

    lr = get_lr(step)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    #torch.cuda.synchronize() #waits for the all kernels on all streams to complete. GPU to catch up to CPU scheduled tasks
    t1 = time.time()
    dt = (t1-t0) * 1000
    
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d}, loss {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f}, dt: {dt*1000:.2f}ms, tok/s: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()

#-------------------------------------------------------
#int8 used for inference not training.

#GPU bandwidth is a bottleneck, your tensor cores are processing maybe 50% of the time

#classifier layer at the top going from 768 to the vocab_size. it is the heaviest process
#TF32 (tensor float 32) is a performant improvement to FP32 when using NVIDIA GPUs

#print(logits.shape)

#iterate x y batches and make a dataloadr to optimise a reasonable objective, with new batches
#num_return_sequences =5
#max_length = 30
# Move the model to MPS (Metal Performance Shaders) for Apple Silicon



# print("Input tensor x:", x)

# torch.manual_seed(42)

