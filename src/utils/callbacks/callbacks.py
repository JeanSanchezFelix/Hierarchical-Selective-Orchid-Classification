import torch
import logging

# TODO: split up each callback into different files? Example: ModelCheckPoint.py, EarlyStopping.py, etc.
class Callback:
    """
    Base class for all callbacks.
    """

    def on_epoch_start(self, epoch: int, logs: dict = None):
        pass

    def on_epoch_end(self, epoch: int, logs: dict = None):
        pass

    def on_train_start(self, logs: dict = None):
        pass

    def on_train_end(self, logs: dict = None):
        pass

    def on_validation_start(self, logs: dict = None):
        pass

    def on_validation_end(self, logs: dict = None):
        pass

class ModelCheckpoint(Callback):
    """
    Saves the model during training whenever a monitored metric improves.

    Attributes:
        monitor (str): Metric to monitor (e.g., 'val_loss').
        save_best_only (bool): If True, only saves the best model.
        mode (str): One of {'min', 'max'} to determine improvement direction.
        save_path (str): File path to save the model checkpoint.
        verbose (bool): If True, logs when a model is saved.
    """
    def __init__(self, monitor='val_loss', save_best_only=True, mode='min', save_path='model_checkpoint.pth', verbose=False):
        if mode not in ['min', 'max']:
            raise ValueError("Mode must be 'min' or 'max'")
        
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_path = save_path
        self.verbose = verbose
        self.best_score = None

    def on_epoch_end(self, epoch: int, logs: dict = None):
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(f"Metric {self.monitor} is not available in logs.")
            return

        if not self.save_best_only:
            self._save_model(logs["model"])
        else:
            if self.best_score is None or self._is_improvement(current):
                self.best_score = current
                self._save_model(logs["model"])

    def _is_improvement(self, current):
        if self.mode == 'min':
            return current < self.best_score
        return current > self.best_score

    def _save_model(self, model):
        torch.save(model.state_dict(), self.save_path)
        if self.verbose:
            logging.info(f"Model saved to {self.save_path}")

class EarlyStopping(Callback):
    """
    Stops training when a monitored metric stops improving.

    Attributes:
        monitor (str): Metric to monitor (e.g., 'val_loss').
        patience (int): Number of epochs to wait before stopping.
        min_delta (float): Minimum change to consider as improvement.
        mode (str): One of {'min', 'max'}.
        save_path (str): File path to save the best model.
        verbose (bool): If True, logs messages when Early Stop is triggered.
    """
    def __init__(self, monitor='val_loss', patience=5, min_delta=0.0, mode='min', save_path='best_model.pth', verbose=False):
        if mode not in ['min', 'max']:
            raise ValueError("Mode must be 'min' or 'max'")
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path
        self.best_score = None
        self.wait = 0
        self.early_stop = False
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: dict = None):
        current = logs.get(self.monitor)
        if current is None:
            if self.verbose:
                logging.warning(f"Metric {self.monitor} is not available in logs.")
            return

        if self.best_score is None:
            self.best_score = current
            self._save_model(logs["model"])
        elif self._is_improvement(current):
            self.best_score = current
            self.wait = 0
            self._save_model(logs["model"])
        else:
            self.wait += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement. Counter {self.wait}/{self.patience}")
            if self.wait >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logging.info(f"Early stopping triggered at epoch {epoch + 1}.")

    def _is_improvement(self, current):
        if self.mode == 'min':
            return current < self.best_score - self.min_delta
        return current > self.best_score + self.min_delta

    def _save_model(self, model):
        torch.save(model.state_dict(), self.save_path)
        if self.verbose:
            logging.info(f"Model saved to {self.save_path}")


class ReduceLROnPlateau(Callback):
    """
    Reduces the learning rate when a monitored metric has stopped improving.

    Attributes:
        monitor (str): Metric to monitor (e.g., 'val_loss').
        factor (float): Factor by which to reduce the learning rate.
        patience (int): Number of epochs with no improvement before reducing the LR.
        min_lr (float): Minimum learning rate.
    """
    def __init__(self, monitor='val_loss', factor=0.1, patience=5, min_delta=0.0, mode='min', min_lr=1e-6, verbose=False):
        if mode not in ['min', 'max']:
            raise ValueError("Mode must be 'min' or 'max'")
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.min_lr = min_lr
        self.best_score = None
        self.wait = 0
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: dict = None):
        current = logs.get(self.monitor)
        optimizer = logs.get("optimizer")
        if current is None or optimizer is None:
            if self.verbose:
                logging.warning(f"Metric {self.monitor} or optimizer is not available in logs.")
            return

        if self.best_score is None:
            self.best_score = current
        elif not self._is_improvement(current):
            self.wait += 1
            if self.verbose:
                print(f"ReduceLROnPlateau: No improvement. Counter {self.wait}/{self.patience}")
            if self.wait >= self.patience:
                self._reduce_lr(optimizer)
                self.wait = 0
        else:
            self.best_score = current
            self.wait = 0

    def _is_improvement(self, current):
        if self.mode == 'min':
            return current < self.best_score - self.min_delta
        return current > self.best_score + self.min_delta

    def _reduce_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            new_lr = max(param_group['lr'] * self.factor, self.min_lr)
            param_group['lr'] = new_lr
        if self.verbose:
            logging.info(f"Reduced learning rate to {new_lr}")