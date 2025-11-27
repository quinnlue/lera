import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import sys
import time
import random
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lera.model.standard_model import Model
from train.utils import Metrics
from tokenizers import Tokenizer
from torch.optim import AdamW
from tqdm import tqdm

_use_amp = True
_use_scaler = True
EPOCHS = 300
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
D_MODEL = 16
N_HEADS = 8
VOCAB_SIZE = 10 #len(tokenizer.get_vocab())
OUTPUT_DIM = 10 # 1001 # [-500, 500]
MAX_SEQ_LEN = 4096
DEPTH = 1 
model = Model(D_MODEL, N_HEADS, VOCAB_SIZE, OUTPUT_DIM, MAX_SEQ_LEN, DEPTH)
optimizer = AdamW(model.parameters(), lr=0.001)
DEVICE = model.device
NUM_PARAMETERS = sum(p.numel() for p in model.parameters())


print(f"Using device: {DEVICE}")
print(f"""
Model configuration:
    - D_MODEL: {D_MODEL}
    - N_HEADS: {N_HEADS}
    - VOCAB_SIZE: {VOCAB_SIZE}
    - OUTPUT_DIM: {OUTPUT_DIM}
    - MAX_SEQ_LEN: {MAX_SEQ_LEN}
    - DEPTH: {DEPTH}
    - Number of parameters: {NUM_PARAMETERS}
""")
print(f"Model: {model}")


class DummyDataset(Dataset):
    def __init__(self, num_samples=320, seq_len=24):
        self.num_samples = num_samples
        self.seq_len = seq_len
        # Generate fixed data once for overfitting test
        self.data = [(torch.randint(0, VOCAB_SIZE, (self.seq_len,)), 
                      torch.randint(0, OUTPUT_DIM, ())) 
                     for _ in range(num_samples)]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

dataset = DummyDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

metrics = Metrics(EPOCHS, len(dataloader), log_frequency=100, training_log_path="logs/training.log")

for epoch in range(EPOCHS):
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{EPOCHS}", dynamic_ncols=True)

    for batch_idx, batch in pbar:
        optimizer.zero_grad()
        
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=_use_amp):
            x, y = batch
            y_hat = model(x)
            loss = F.cross_entropy(
                y_hat, y, reduction="mean"
            )
        
        predictions = torch.argmax(y_hat, dim=-1)
        correct = (predictions == y).float()
        batch_accuracy = correct.mean().item()
        metrics.update(loss.item(), batch_accuracy, optimizer, pbar)
        
        loss.backward()
        optimizer.step()



