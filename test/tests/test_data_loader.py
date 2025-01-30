import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import DataLoaderLite, load_tokens
from model import GPTConfig

def test_load_and_batch():
    B = 2
    T = 32
    process_rank = 0
    num_processes = 1
    split = "train"

    # Initialize DataLoaderLite
    loader = DataLoaderLite(B=B, T=T, process_rank=process_rank, num_processes=num_processes, split=split)

    # Load a single batch
    x, y = loader.next_batch()

    # Print batch shapes
    print(f"Input batch shape: {x.shape}")      # Expected: (B, T)
    print(f"Target batch shape: {y.shape}")     # Expected: (B, T)

    # Print first input and target sequences
    print("First input sequence:", x[0])
    print("First target sequence:", y[0])


if __name__ == "__main__":
    test_load_and_batch()

""""
cd ./tests
python test_data_loader.py
"""
