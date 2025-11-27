import torch
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lera.model.standard_model import Model
from train.utils import Metrics, LRScheduler
from data.parquet_dataloader import get_dataloader
from tokenizers import Tokenizer
from torch.optim import AdamW
from tqdm import tqdm

_use_amp = True
_use_scaler = True
EPOCHS = 3
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
D_MODEL = 1024
N_HEADS = 8
VOCAB_SIZE = len(tokenizer.get_vocab())
OUTPUT_DIM = 1001 # 1001 # [-500, 500]
MAX_SEQ_LEN = 2048
DEPTH = 24
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

# Load bucketed parquet dataloader
dataloader = get_dataloader(
    data_dir="data/pretraining/tokenized",
    batch_size=32,
    shuffle=True,
    device=DEVICE,
    drop_last=False
)

# Learning rate scheduler configuration
TOTAL_STEPS = EPOCHS * len(dataloader)
WARMUP_STEPS = int(0.1 * TOTAL_STEPS)  # 10% warmup
MAX_LR = 0.001
MIN_LR = 0.0001
scheduler = LRScheduler(
    optimizer=optimizer,
    max_lr=MAX_LR,
    total_steps=TOTAL_STEPS,
    warmup_steps=WARMUP_STEPS,
    min_lr=MIN_LR
)

metrics = Metrics(EPOCHS, len(dataloader), log_frequency=100, training_log_path="logs/training.log")

for epoch in range(EPOCHS):
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{EPOCHS}", dynamic_ncols=True)

    for batch_idx, batch in pbar:
        optimizer.zero_grad()
        
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=_use_amp):
            input_ids, attention_mask, answer_token = batch
            y_hat = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(
                y_hat, answer_token, reduction="mean"
            )
        loss.backward()
        optimizer.step()
        scheduler.step()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        predictions = torch.argmax(y_hat, dim=-1)
        correct = (predictions == answer_token).float()
        batch_accuracy = correct.mean().item()
        metrics.update(loss.item(), batch_accuracy, optimizer, pbar, scheduler)
        



