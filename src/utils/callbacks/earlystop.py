import torch
import os

class EarlyStopping:
    """
    Implements early stopping to terminate training when performance stops improving on monitored metrics.
    Supports monitoring multiple metrics and restoring from checkpoints.
    """
    def __init__(self, 
                 monitor='loss', 
                 mode='min', 
                 patience=5, 
                 min_delta=0.0, 
                 restore_best_weights=True, 
                 save_path=None, 
                 verbose=False):
        """
        Initialize the early stopping instance.

        Parameters:
            monitor (str): The primary metric to monitor (e.g., 'loss' or 'accuracy').
            mode (str): Whether to minimize ('min') or maximize ('max') the monitored metric.
            patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            restore_best_weights (bool): Whether to restore the best model weights at the end of training.
            save_path (str): Path to save the best model's checkpoint.
            verbose (bool): If True, logs messages when the model improves.
        """
        if mode not in ['min', 'max']:
            raise ValueError("Mode must be 'min' or 'max'")
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.save_path = save_path
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

        # Determine the comparison operator based on the mode
        self._is_improvement = (
            (lambda current, best: current < best - self.min_delta)  # Minimize
            if self.mode == 'min' else
            (lambda current, best: current > best + self.min_delta)  # Maximize
        )

    def __call__(self, metrics, model=None, optimizer=None, scheduler=None):
        """
        Check if training should stop based on the monitored metrics.

        Parameters:
            metrics (dict): A dictionary of metrics (e.g., {'loss': 0.4, 'accuracy': 0.85}).
            model (torch.nn.Module): The model to save if the monitored metric improves.
            optimizer (torch.optim.Optimizer): The optimizer to save if the monitored metric improves.
            scheduler (torch.optim.lr_scheduler): The scheduler to save if the monitored metric improves.
        """
        # Get the value of the primary monitored metric
        metric_value = metrics.get(self.monitor)
        if metric_value is None:
            raise ValueError(f"Monitored metric '{self.monitor}' not found in metrics: {metrics.keys()}")

        # Check for improvement
        improvement = False
        if self.best_score is None:
            self.best_score = metric_value
            improvement = True
        elif self._is_improvement(metric_value, self.best_score):
            improvement = True

        # Update state
        if improvement:
            self.best_score = metric_value
            self._save_checkpoint(model, optimizer, scheduler, metrics)
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: Improvement found. Best {self.monitor}: {self.best_score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement. Counter {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        # Log secondary metrics
        if self.verbose:
            for key, value in metrics.items():
                if key != self.monitor:
                    print(f"EarlyStopping: {key.capitalize()}: {value:.4f}")

    def _save_checkpoint(self, model, optimizer, scheduler, metrics):
        """
        Saves the model, optimizer, scheduler state_dict, and metrics if a save path is specified.

        Parameters:
            model (torch.nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer to save.
            scheduler (torch.optim.lr_scheduler): The scheduler to save.
            metrics (dict): The metrics to save.
        """
        if self.save_path:
            checkpoint = {
                'model_state_dict': model.state_dict() if model else None,
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'metrics': metrics,
                'best_score': self.best_score
            }
            torch.save(checkpoint, self.save_path)
            if self.verbose:
                print(f"EarlyStopping: Checkpoint saved to {self.save_path}")

    def restore_checkpoint(self, model=None, optimizer=None, scheduler=None):
        """
        Restores the model, optimizer, scheduler state_dict, and metrics from the checkpoint.

        Parameters:
            model (torch.nn.Module): The model to restore.
            optimizer (torch.optim.Optimizer): The optimizer to restore.
            scheduler (torch.optim.lr_scheduler): The scheduler to restore.
        """
        if not self.save_path or not os.path.exists(self.save_path):
            raise ValueError("No checkpoint file found to restore.")

        checkpoint = torch.load(self.save_path)
        if model and checkpoint['model_state_dict']:
            model.load_state_dict(checkpoint['model_state_dict'])
            if self.verbose:
                print(f"EarlyStopping: Model restored from {self.save_path}")
        if optimizer and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.verbose:
                print(f"EarlyStopping: Optimizer restored from {self.save_path}")
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.verbose:
                print(f"EarlyStopping: Scheduler restored from {self.save_path}")
        self.best_score = checkpoint.get('best_score', self.best_score)

    def reset(self):
        """
        Resets the early stopping parameters.
        """
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None
