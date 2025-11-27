import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import pandas as pd
import ast
import numpy as np
from lera.model.model import RRMWithLoadBalancing



df = pd.read_csv("prototyping/data/fol_train_64k_depth_6.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X = torch.tensor(df['seq'].apply(ast.literal_eval).tolist())
Y = torch.tensor(df['result'].tolist())

# Create attention mask [B, 1, S, S] - True where tokens can attend
padding_mask = (X != 3)  # [B, S] - True where NOT padding
attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
attn_mask = attn_mask.expand(-1, -1, X.shape[1], -1)  # [B, 1, S, S]

# Split data into train and validation (90/10 split)
total_samples = len(X)
train_size = int(0.9 * total_samples)
val_size = total_samples - train_size

# Create indices for splitting
indices = torch.randperm(total_samples)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Split the data
X_train, X_val = X[train_indices], X[val_indices]
Y_train, Y_val = Y[train_indices], Y[val_indices]
attn_mask_train = attn_mask[train_indices]
attn_mask_val = attn_mask[val_indices]

# Create dataset and dataloader
class MultiOpsDataset(Dataset):
    def __init__(self, X, Y, attn_mask):
        self.X = X
        self.Y = Y
        self.attn_mask = attn_mask
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.attn_mask[idx]

train_dataset = MultiOpsDataset(X_train, Y_train, attn_mask_train)
val_dataset = MultiOpsDataset(X_val, Y_val, attn_mask_val)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

print(f"Created dataloaders with batch size 256")
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")
print(f"Input sequence shape: {X.shape}")
print(f"Target shape: {Y.shape}")
print(f"Attention mask shape: {attn_mask.shape}")

tok2idx = {'<pad>': 0,
 '<router>': 1,
 '<fact_vector>': 2,
 '<temperature>': 3,
 '<begin_world>': 4,
 '<end_world>': 5,
 '(': 6,
 ')': 7,
 ',': 8,
 '0': 9,
 '1': 10,
 '=': 11,
 'And': 12,
 'Exists': 13,
 'F': 14,
 'ForAll': 15,
 'Implies': 16,
 'Not': 17,
 'Or': 18,
 'P': 19,
 'Q': 20,
 'R': 21,
 'T': 22,
 'x': 23}

vocab_size = len(tok2idx)

print(f"Vocab size: {vocab_size}")
print(f"Expected starting loss: {np.log(vocab_size)}")

# Model hyperparameters
D_MODEL = 64
N_HEADS = 4
N_RECURSIONS = 1
NUM_EXPERTS = 1
TEMPERATURE_IDX = 1
ROUTER_IDX = 0  # Position in sequence where router makes decisions
FACT_VECTOR_IDX = 2
PREPROCESS_DEPTH = 4
LOAD_BALANCE_COEFF = 0.01  # Alpha parameter for load balancing


print("Model Configuration:")
print(f"  d_model: {D_MODEL}")
print(f"  n_heads: {N_HEADS}")
print(f"  recursions: {N_RECURSIONS}")
print(f"  num_experts: {NUM_EXPERTS}")
print(f"  router_idx: {ROUTER_IDX}")
print(f"  fact_vector_idx: {FACT_VECTOR_IDX}")
print(f"  load_balance_coeff: {LOAD_BALANCE_COEFF}")



# Get actual sequence length from data
CONTEXT_LEN = X.shape[1]  # Should be 16 based on the data
print(f"Sequence length (context_len): {CONTEXT_LEN}")

# Create model with load balancing
model = RRMWithLoadBalancing(
    d_model=D_MODEL,
    n_heads=N_HEADS,
    max_recursions=N_RECURSIONS,
    vocab_size=vocab_size,
    num_experts=NUM_EXPERTS,
    router_idx=ROUTER_IDX,
    fact_vector_idx=FACT_VECTOR_IDX,
    temperature_idx=TEMPERATURE_IDX,
    preprocess_depth=PREPROCESS_DEPTH,
    context_len=CONTEXT_LEN,
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel created!")
print(f"Total parameters: {total_params:,}")

# Create optimizer
optimizer = AdamW(model.parameters(), lr=0.001)
print("Optimizer created with lr=0.001")

EPOCHS = 2
TOTAL_STEPS = len(train_loader) * EPOCHS
WARMUP_STEPS = 100

warmup_scheduler = LinearLR(
    optimizer, 
    start_factor=0.01,
    end_factor=1.0,
    total_iters=WARMUP_STEPS
)

cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=TOTAL_STEPS - WARMUP_STEPS,
    eta_min=1e-6
)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[WARMUP_STEPS]
)

print(f"Scheduler created: {WARMUP_STEPS} warmup steps, then cosine annealing to step {TOTAL_STEPS}")
from tqdm import tqdm
import torch.nn.functional as F
from collections import deque
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from train.utils import setup_logger

model.to(device)
model.to(torch.float32)
model.train()

task_losses = []
lb_losses = []
total_losses = []
accuracies = []

# Track last 10 batches for moving averages
recent_batch_accuracies = deque(maxlen=10)
recent_batch_losses = deque(maxlen=10)

# Setup logging for training metrics
training_logger = setup_logger('prototype_training', 'prototype_training.log', header="batch,task_loss,lb_loss,total_loss,accuracy")

print("Starting training...\n")

for epoch in range(EPOCHS):
    print(f"=== Epoch {epoch + 1}/{EPOCHS} ===")
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True)
    
    for batch_idx, (batch_x, batch_y, batch_mask) in pbar:
        # Move batch to GPU right before computation
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_mask = batch_mask.to(device)
        
        x_emb = model.encoder(batch_x)
        
        y_hat, lb_loss, deep_supervision_loss = model.forward(x_emb, context_length=CONTEXT_LEN, context_mask=None, attn_mask=batch_mask, return_load_balance_loss=True, return_deep_supervision_loss=True, y_true=batch_y)
        
        # Calculate accuracy
        predictions = torch.argmax(y_hat, dim=-1)
        correct = (predictions == batch_y).float()
        batch_accuracy = correct.mean().item()
        
        # Task loss (cross entropy on final output)
        loss = F.cross_entropy(
            input=y_hat,
            target=batch_y,
            reduction="mean"
        )
        
        # Total loss: main task loss + load balancing + averaged deep supervision
        # deep_supervision_loss is already accumulated over N_RECURSIONS steps, so we average it
        total_loss = loss + LOAD_BALANCE_COEFF * lb_loss + (deep_supervision_loss / N_RECURSIONS)
        
        # Backprop
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate
        
        # Track losses and accuracy
        task_losses.append(loss.item())
        lb_losses.append(lb_loss.item())
        total_losses.append(total_loss.item())
        accuracies.append(batch_accuracy)
        
        # Log metrics to file
        global_step = epoch * len(train_loader) + batch_idx
        training_logger.info(f"{global_step},{loss.item():.6f},{lb_loss.item():.6f},{total_loss.item():.6f},{batch_accuracy:.6f}")
        
        # Update moving averages (per-batch tracking)
        recent_batch_accuracies.append(batch_accuracy)
        recent_batch_losses.append(loss.item())
        
        # Calculate moving averages over last 10 batches
        moving_avg_acc = sum(recent_batch_accuracies) / len(recent_batch_accuracies) if recent_batch_accuracies else 0.0
        moving_avg_loss = sum(recent_batch_losses) / len(recent_batch_losses) if recent_batch_losses else 0.0
        
        # Update tqdm bar with current metrics and moving averages
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_description(f"Batch {batch_idx+1}/{len(train_loader)}")
        pbar.set_postfix({
            "Acc": f"{batch_accuracy:.3f}",
            "MA_Acc": f"{moving_avg_acc:.3f}",
            "Loss": f"{loss.item():.3f}",
            "MA_Loss": f"{moving_avg_loss:.3f}",
            "LB": f"{lb_loss.item():.2f}",
            "LR": f"{current_lr:.2e}"
        })

print("Training complete!")

# Validation loop
print("\n=== Running Validation ===")
model.eval()
val_losses = []
val_accuracies = []
val_lb_losses = []

# Setup validation logger
val_logger = setup_logger('prototype_validation', 'prototype_validation.log', header="batch,loss,accuracy,lb_loss")

with torch.no_grad():
    val_pbar = tqdm(val_loader, total=len(val_loader), dynamic_ncols=True, desc="Validation")
    
    for batch_x, batch_y, batch_mask in val_pbar:
        # Move batch to GPU right before computation
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_mask = batch_mask.to(device)
        
        x_emb = model.encoder(batch_x)
        
        y_hat, lb_loss, deep_supervision_loss = model.forward(
            x_emb, 
            context_length=CONTEXT_LEN, 
            context_mask=None, 
            attn_mask=batch_mask, 
            return_load_balance_loss=True, 
            return_deep_supervision_loss=True, 
            y_true=batch_y
        )
        
        # Calculate accuracy
        predictions = torch.argmax(y_hat, dim=-1)
        correct = (predictions == batch_y).float()
        batch_accuracy = correct.mean().item()
        
        # Calculate loss
        loss = F.cross_entropy(
            input=y_hat,
            target=batch_y,
            reduction="mean"
        )
        
        val_losses.append(loss.item())
        val_accuracies.append(batch_accuracy)
        val_lb_losses.append(lb_loss.item())
        
        # Log validation metrics
        val_logger.info(f"{len(val_losses)-1},{loss.item():.6f},{batch_accuracy:.6f},{lb_loss.item():.6f}")
        
        val_pbar.set_postfix({
            "Loss": f"{loss.item():.3f}",
            "Acc": f"{batch_accuracy:.3f}"
        })

# Calculate validation metrics
val_loss_mean = np.mean(val_losses)
val_acc_mean = np.mean(val_accuracies)
val_lb_mean = np.mean(val_lb_losses)

print(f"\nValidation Results:")
print(f"  Loss: {val_loss_mean:.4f}")
print(f"  Accuracy: {val_acc_mean:.4f}")
print(f"  LB Loss: {val_lb_mean:.4f}")

# Log final summary
summary_logger = setup_logger('prototype_summary', 'prototype_summary.log', header="=== Training Summary ===")
summary_logger.info(f"Final Training Loss: {task_losses[-1]:.6f}")
summary_logger.info(f"Final Training Accuracy: {accuracies[-1]:.6f}")
summary_logger.info(f"Final Training LB Loss: {lb_losses[-1]:.6f}")
summary_logger.info(f"Final Training Total Loss: {total_losses[-1]:.6f}")
summary_logger.info("")
summary_logger.info("=== Validation Summary ===")
summary_logger.info(f"Mean Validation Loss: {val_loss_mean:.6f}")
summary_logger.info(f"Mean Validation Accuracy: {val_acc_mean:.6f}")
summary_logger.info(f"Mean Validation LB Loss: {val_lb_mean:.6f}")

print(f"\nTraining metrics saved to prototype_training.log")
print(f"Validation metrics saved to prototype_validation.log")
print(f"Summary saved to prototype_summary.log")
print(f"\nFinal Training - Loss: {task_losses[-1]:.4f}, Accuracy: {accuracies[-1]:.4f}")
print(f"Final Validation - Loss: {val_loss_mean:.4f}, Accuracy: {val_acc_mean:.4f}")
