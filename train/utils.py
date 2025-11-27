import collections
import time
import os
import warnings
import logging
import math
from typing import Tuple, Optional

class RunningLossTracker:
    def __init__(self, window_sizes: Tuple[int, int, int] = (100, 1000, 10000)):
        self.window_100, self.window_1k, self.window_10k = window_sizes
        
        self.loss_window_100 = collections.deque(maxlen=self.window_100)
        self.loss_window_1k = collections.deque(maxlen=self.window_1k)
        self.loss_window_10k = collections.deque(maxlen=self.window_10k)
        
        self.sum_100 = 0.0
        self.sum_1k = 0.0
        self.sum_10k = 0.0
        
        self.count_100 = 0
        self.count_1k = 0
        self.count_10k = 0
    
    def update(self, loss_value: float):
        if len(self.loss_window_100) == self.window_100:
            self.sum_100 -= self.loss_window_100[0]
        else:
            self.count_100 += 1
        
        self.loss_window_100.append(loss_value)
        self.sum_100 += loss_value
        
        if len(self.loss_window_1k) == self.window_1k:
            self.sum_1k -= self.loss_window_1k[0]
        else:
            self.count_1k += 1
        
        self.loss_window_1k.append(loss_value)
        self.sum_1k += loss_value
        
        if len(self.loss_window_10k) == self.window_10k:
            self.sum_10k -= self.loss_window_10k[0]
        else:
            self.count_10k += 1
        
        self.loss_window_10k.append(loss_value)
        self.sum_10k += loss_value
    
    def get_running_losses(self) -> Tuple[float, float, float]:
        running_100_loss = self.sum_100 / self.count_100 if self.count_100 > 0 else float('inf')
        running_1k_loss = self.sum_1k / self.count_1k if self.count_1k > 0 else float('inf')
        running_10k_loss = self.sum_10k / self.count_10k if self.count_10k > 0 else float('inf')
        
        return running_100_loss, running_1k_loss, running_10k_loss

def format_fraction(current: int, total: int) -> str:
    if total == 0:
        return "0.000%"
    percentage = (current / total) * 100
    return f"{percentage:.3f}%"

def calculate_steps_per_sec(current_step: int, start_time: float) -> float:
    elapsed_time = time.perf_counter() - start_time
    return current_step / elapsed_time if elapsed_time > 0 else 0.0

def setup_logger(name: str, log_file: str, mode: str = 'w', header: Optional[str] = None) -> logging.Logger:
    """Setup a logger with file handler in one line.
    
    Args:
        name: Logger name
        log_file: Path to log file
        mode: File mode ('w' for overwrite, 'a' for append)
        header: Optional header line to write first
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    if header:
        logger.info(header)
    
    return logger

class ProgressBarManager:
    def __init__(self, total_steps: int, start_time: float):
        self.total_steps = total_steps
        self.start_time = start_time
        self.current_step = 0
        self.loss_tracker = RunningLossTracker()
        self.accuracy_tracker = RunningLossTracker()
    
    def update_progress(self, loss_value: float, accuracy_value, optimizer, pbar, scheduler=None) -> None:
        self.current_step += 1
        self.loss_tracker.update(loss_value)
        self.accuracy_tracker.update(accuracy_value)
        
        running_100_loss, running_1k_loss, running_10k_loss = self.loss_tracker.get_running_losses()
        running_100_accuracy, running_1k_accuracy, running_10k_accuracy = self.accuracy_tracker.get_running_losses()
        steps_per_sec = calculate_steps_per_sec(self.current_step, self.start_time)
        fraction = format_fraction(self.current_step, self.total_steps)
        
        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        pbar.set_postfix(
            L100=f"{running_100_loss:.4f}",
            L1k=f"{running_1k_loss:.4f}",
            L10k=f"{running_10k_loss:.4f}",
            A100=f"{running_100_accuracy:.4f}",
            A1k=f"{running_1k_accuracy:.4f}",
            A10k=f"{running_10k_accuracy:.4f}",
            LR=f"{current_lr:.6f}",
        )
        pbar.update(1)


class Metrics:
    def __init__(self, num_epochs, dataloader_length, log_frequency=100,
                 training_log_path="training.log", validation_log_path="validation.log"):
        if os.path.exists(training_log_path):
            warnings.warn(f"Training log file {training_log_path} already exists. Writing to the end of the file.")
        self.num_epochs = num_epochs
        self.dataloader_length = dataloader_length
        self.total_steps = num_epochs * dataloader_length
        self.log_frequency = log_frequency
        
        self.training_log_path = training_log_path
        self.validation_log_path = validation_log_path
        
        # Setup logger for training
        self.logger = setup_logger('training_metrics', training_log_path, mode='a', header="step,loss,accuracy") if training_log_path else None
        
        self.row_buffer = []
        self.pbar_manager = ProgressBarManager(self.total_steps, time.perf_counter())

    def update(self, loss, accuracy, optimizer, pbar, scheduler=None):
        self.pbar_manager.update_progress(loss, accuracy, optimizer, pbar, scheduler)

        self.row_buffer.append({
            "step": self.pbar_manager.current_step,
            "loss": loss,
            "accuracy": accuracy
        })

        if self.pbar_manager.current_step % self.log_frequency == 0:
            if self.row_buffer and self.logger:
                # Write buffered rows to log file
                for row in self.row_buffer:
                    self.logger.info(f"{row['step']},{row['loss']:.6f},{row['accuracy']:.6f}")
                self.row_buffer.clear()


class LRScheduler:
    """Learning rate scheduler with warmup and cosine annealing.
    
    Supports:
    - Linear warmup from 0 to max_lr over warmup_steps
    - Cosine annealing from max_lr to min_lr over total_steps - warmup_steps
    """
    
    def __init__(
        self,
        optimizer,
        max_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """Initialize the learning rate scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            max_lr: Maximum learning rate (peak after warmup)
            total_steps: Total number of training steps
            warmup_steps: Number of warmup steps (linear increase from 0 to max_lr)
            min_lr: Minimum learning rate (final value after cosine decay)
            last_epoch: The index of last epoch (for resuming training)
        """
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Initialize learning rates
        self._update_lr(0)
    
    def _update_lr(self, step: int):
        """Update learning rate based on current step."""
        if step < self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * (step / max(self.warmup_steps, 1))
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def step(self, epoch=None):
        """Update learning rate for the next step."""
        self.last_epoch += 1
        self._update_lr(self.last_epoch)
    
    def get_last_lr(self):
        """Get the last computed learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Return state dict for checkpointing."""
        return {
            'last_epoch': self.last_epoch,
            'max_lr': self.max_lr,
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps,
            'min_lr': self.min_lr,
            'base_lrs': self.base_lrs,
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.last_epoch = state_dict['last_epoch']
        self.max_lr = state_dict['max_lr']
        self.total_steps = state_dict['total_steps']
        self.warmup_steps = state_dict['warmup_steps']
        self.min_lr = state_dict['min_lr']
        self.base_lrs = state_dict['base_lrs']
        self._update_lr(self.last_epoch)