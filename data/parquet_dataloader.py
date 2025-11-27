import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import glob
import os
from typing import List, Tuple
import numpy as np


class BucketedParquetDataset(Dataset):
    """
    Dataset that loads tokenized parquet files and groups samples by bucket_id.
    Sequences are already padded to their bucket's sequence length.
    """
    
    def __init__(self, data_dir: str, device: torch.device = None):
        """
        Args:
            data_dir: Directory containing train-*.parquet files
            device: Device to place tensors on (cuda/cpu)
        """
        self.device = device if device is not None else torch.device('cpu')
        
        # Auto-discover all parquet files
        parquet_pattern = os.path.join(data_dir, "train-*.parquet")
        parquet_files = sorted(glob.glob(parquet_pattern))
        
        if len(parquet_files) == 0:
            raise ValueError(f"No parquet files found matching pattern: {parquet_pattern}")
        
        print(f"Loading {len(parquet_files)} parquet file(s) from {data_dir}")
        
        # Load and concatenate all parquet files
        dfs = []
        for file in parquet_files:
            df = pd.read_parquet(file)
            dfs.append(df)
            print(f"  - Loaded {file}: {len(df)} samples")
        
        self.data = pd.concat(dfs, ignore_index=True)
        print(f"Total samples loaded: {len(self.data)}")
        
        # Group indices by bucket_id for efficient sampling
        self.bucket_indices = {}
        for bucket_id in self.data['bucket_id'].unique():
            indices = self.data[self.data['bucket_id'] == bucket_id].index.tolist()
            self.bucket_indices[bucket_id] = indices
        
        print(f"Dataset organized into {len(self.bucket_indices)} buckets:")
        for bucket_id in sorted(self.bucket_indices.keys()):
            print(f"  - Bucket {bucket_id}: {len(self.bucket_indices[bucket_id])} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_ids: Tensor of shape (seq_len,)
            attention_mask: Tensor of shape (seq_len,)
            answer_token: Tensor (scalar)
        """
        row = self.data.iloc[idx]
        
        # Convert to tensors
        input_ids = torch.tensor(row['input_ids'], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(row['attention_mask'], dtype=torch.long, device=self.device)
        answer_token = torch.tensor(row['answer_token'], dtype=torch.long, device=self.device)
        
        return input_ids, attention_mask, answer_token
    
    def get_bucket_indices(self):
        """Returns dictionary mapping bucket_id to list of sample indices."""
        return self.bucket_indices


class BucketedBatchSampler(Sampler):
    """
    Custom sampler that yields batches where all samples come from the same bucket.
    Shuffles within each bucket independently at each epoch.
    """
    
    def __init__(self, dataset: BucketedParquetDataset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        """
        Args:
            dataset: BucketedParquetDataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle within buckets
            drop_last: Whether to drop the last incomplete batch of each bucket
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_indices = dataset.get_bucket_indices()
        
    def __iter__(self):
        # Prepare shuffled indices for each bucket
        bucket_batches = []
        
        for bucket_id in sorted(self.bucket_indices.keys()):
            indices = self.bucket_indices[bucket_id].copy()
            
            # Shuffle within bucket if requested
            if self.shuffle:
                np.random.shuffle(indices)
            
            # Create batches from this bucket
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                
                # Skip incomplete batches if drop_last is True
                if len(batch) == self.batch_size or not self.drop_last:
                    bucket_batches.append(batch)
        
        # Shuffle the order of batches across buckets
        if self.shuffle:
            np.random.shuffle(bucket_batches)
        
        # Yield batches
        for batch in bucket_batches:
            yield batch
    
    def __len__(self):
        total_batches = 0
        for bucket_id, indices in self.bucket_indices.items():
            n_samples = len(indices)
            if self.drop_last:
                total_batches += n_samples // self.batch_size
            else:
                total_batches += (n_samples + self.batch_size - 1) // self.batch_size
        return total_batches


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function to stack tensors into batches.
    Since sequences within a bucket are already the same length, no padding needed.
    
    Args:
        batch: List of (input_ids, attention_mask, answer_token) tuples
    
    Returns:
        input_ids: Tensor of shape (batch_size, seq_len)
        attention_mask: Tensor of shape (batch_size, seq_len)
        answer_tokens: Tensor of shape (batch_size,)
    """
    input_ids = torch.stack([item[0] for item in batch])
    attention_masks = torch.stack([item[1] for item in batch])
    answer_tokens = torch.stack([item[2] for item in batch])
    
    return input_ids, attention_masks, answer_tokens


def get_dataloader(
    data_dir: str = "data/pretraining/tokenized",
    batch_size: int = 32,
    shuffle: bool = True,
    device: torch.device = None,
    drop_last: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """
    Factory function to create a DataLoader for bucketed parquet data.
    
    Args:
        data_dir: Directory containing train-*.parquet files
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data (within buckets)
        device: Device to place tensors on
        drop_last: Whether to drop incomplete batches
        num_workers: Number of workers for data loading (keep 0 for device placement in __getitem__)
    
    Returns:
        DataLoader instance configured for bucketed batching
    """
    dataset = BucketedParquetDataset(data_dir=data_dir, device=device)
    sampler = BucketedBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False  # Already on device in __getitem__
    )
    
    return dataloader

