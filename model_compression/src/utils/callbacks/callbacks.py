import os
import logging
from typing import Dict, Any
import torch

class Callback:
    """
    Base class for training callbacks with hooks at various stages.
    """
    def on_train_start(self, logs: Dict[str, Any] = None) -> None:
        """Called at the beginning of training."""
        pass

    def on_epoch_start(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_validation_start(self, logs: Dict[str, Any] = None) -> None:
        """Called at the beginning of validation."""
        pass

    def on_validation_end(self, logs: Dict[str, Any] = None) -> None:
        """Called at the end of validation."""
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Called at the end of each epoch."""
        pass

    def on_train_end(self, logs: Dict[str, Any] = None) -> None:
        """Called at the end of training."""
        pass

class ModelCheckpoint(Callback):
    """
    Save the model checkpoint when a monitored metric improves.

    Attributes:
        monitor: Key in logs to monitor (e.g., 'val_loss').
        save_best_only: If True, only saves when metric improves.
        mode: 'min' or 'max' to interpret improvement.
        save_path: File path for saving model state_dict.
        verbose: If True, logs save events.
    """
    def __init__(
        self,
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        mode: str = 'min',
        save_path: str = 'model_checkpoint.pth',
        verbose: bool = False
    ) -> None:
        if mode not in ('min', 'max'):
            raise ValueError("mode must be 'min' or 'max'")
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_path = save_path
        self.verbose = verbose
        self.best_score: float = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Check for improvement and save model state."""
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(f"Metric {self.monitor} is not available in logs.")
            return  
        
        improvement = self._is_improvement(current)
        if not self.save_best_only or improvement:
            if improvement:
                self.best_score = current  # update only on improvement
            self._save_model(epoch, logs)

    def _is_improvement(self, current: float):
        if self.mode == 'min':
            return current < self.best_score
        return current > self.best_score

    def _save_model(self, epoch: int, logs: Dict[str, Any]):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(logs["model"].state_dict(), self.save_path)
        # Save metadata
        meta = {
            'epoch': epoch,
            'learning_rate': logs.get('learning_rate'),
            'batch_size': logs.get('batch_size'),
            'criterion': logs.get('criterion'),
            'optimizer': logs.get('optimizer').state_dict() if logs.get('optimizer') else None
        }
        meta_path = os.path.join(os.path.dirname(self.save_path), 'metadata.pth')
        torch.save(meta, meta_path)
        if self.verbose:
            logging.info(f"Checkpoint saved at epoch {epoch + 1} to {self.save_path}")

class EarlyStopping(Callback):
    """
    Stop training when monitored metric stops improving.

    Attributes:
        monitor: Metric name to monitor.
        patience: Epochs to wait after last improvement.
        min_delta: Minimum change to qualify as improvement.
        mode: 'min' or 'max'.
        save_path: Path to save best model.
        verbose: If True, logs early stopping events.
    """
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
        save_path: str = 'best_model.pth',
        verbose: bool = False
    ) -> None:
        if mode not in ('min', 'max'):
            raise ValueError("mode must be 'min' or 'max'")
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path
        self.verbose = verbose
        self.best_score: float = float('inf') if mode == 'min' else float('-inf')
        self.wait: int = 0
        self.early_stop: bool = False

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Check for early stopping condition and save best model."""
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(f"{self.monitor} not found in logs.")
            return
        
        improvement= self._is_improvement(current)
        if improvement:
            self.best_score = current
            self.wait = 0
            # Save best model
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(logs['model'].state_dict(), self.save_path)
            if self.verbose:
                logging.info(f"EarlyStopping: Improvement detected, model saved to {self.save_path}")
        else:
            self.wait += 1
            if self.verbose:
                logging.info(f"EarlyStopping: {self.wait}/{self.patience} no improvement")
            if self.wait >= self.patience:
                self.early_stop = True
                logging.info(f"EarlyStopping triggered at epoch {epoch + 1}")

    def _is_improvement(self, current: float):
        if self.mode == 'min':
            return current < self.best_score - self.min_delta
        return current > self.best_score + self.min_delta

    def _save_model(self, epoch: int, logs: Dict[str, Any]):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(logs["model"].state_dict(), self.save_path)
        # Save metadata
        meta = {
            'epoch': epoch,
            'learning_rate': logs.get('learning_rate'),
            'batch_size': logs.get('batch_size'),
            'criterion': logs.get('criterion'),
            'optimizer': logs.get('optimizer').state_dict() if logs.get('optimizer') else None
        }
        meta_path = os.path.join(os.path.dirname(self.save_path), 'metadata.pth')
        torch.save(meta, meta_path)
        if self.verbose:
            logging.info(f"Checkpoint saved at epoch {epoch + 1} to {self.save_path}")


class ReduceLROnPlateau(Callback):
    """
    Reduce learning rate when a metric has plateaued.

    Attributes:
        monitor: Metric name to monitor.
        factor: Multiplicative factor of LR reduction.
        patience: Epochs to wait after last improvement.
        min_delta: Minimum change to qualify as improvement.
        mode: 'min' or 'max'.
        min_lr: Lower bound on learning rate.
        verbose: If True, logs LR changes.
    """
    def __init__(
        self,
        monitor: str = 'val_loss',
        factor: float = 0.1,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
        min_lr: float = 1e-6,
        verbose: bool = False
    ) -> None:
        if mode not in ('min', 'max'):
            raise ValueError("mode must be 'min' or 'max'")
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_score: float = float('inf') if mode == 'min' else float('-inf')
        self.wait: int = 0

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """Check for plateau and reduce LR if needed."""
        logs = logs or {}
        current = logs.get(self.monitor)
        optimizer = logs.get('optimizer')
        if current is None or optimizer is None:
            logging.warning(f"{self.monitor} or optimizer not in logs.")
            return

        improvement= self._is_improvement(current)
        if improvement:
            self.best_score = current
            self.wait = 0
        else:
            self.wait += 1
            if self.verbose:
                logging.info(f"ReduceLROnPlateau: {self.wait}/{self.patience} no improvement")
            if self.wait >= self.patience:
                self._reduce_lr(optimizer)
                self.wait = 0

    def _is_improvement(self, current: float):
        if self.mode == 'min':
            return current < self.best_score - self.min_delta
        return current > self.best_score + self.min_delta

    def _reduce_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            old_lr = param_group.get('lr', 0.0)
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
        if self.verbose:
            logging.info(f"LR reduced from {old_lr:.6f} to {new_lr:.6f}")

### TODO: Test if this works and add to registry ###
class LRScheduler(Callback):
    """
    Wrap a PyTorch learning-rate scheduler.

    Attributes:
        scheduler: Instance of torch.optim.lr_scheduler._LRScheduler or similar.
        verbose: If True, logs LR updates.
    """
    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler, verbose: bool = False) -> None:
        self.scheduler = scheduler
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Step the scheduler and optionally log the new LR values."""
        old_lrs = [g['lr'] for g in self.scheduler.optimizer.param_groups]
        # Some schedulers require a metric
        try:
            self.scheduler.step(logs.get(self.scheduler.monitor, None) if hasattr(self.scheduler, 'monitor') else None)
        except TypeError:
            self.scheduler.step()
        new_lrs = [g['lr'] for g in self.scheduler.optimizer.param_groups]
        if self.verbose:
            logging.info(f"LRs updated from {old_lrs} to {new_lrs}")