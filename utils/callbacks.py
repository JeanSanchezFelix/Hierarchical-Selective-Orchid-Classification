import torch
import os

# TODO: Test if it works

class EarlyStopping:
    """
    Implements early stopping to terminate training when validation performance stops improving.
    """
    def __init__(self, patience=5, min_delta=0.0, save_path=None, verbose=False):
        """
        Initialize the early stopping instance.

        Parameters:
            patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            save_path (str): Path to save the best model's state_dict. If None, model is not saved.
            verbose (bool): If True, logs messages when the model improves.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, metric, model=None):
        """
        Check if training should stop based on the monitored metric.

        Parameters:
            metric (float): The current value of the monitored metric (e.g., validation loss).
            model (torch.nn.Module): The model to save if the metric improves.
        """
        if self.best_score is None:
            # First call initializes the best score
            self.best_score = metric
            self._save_model(model)
        elif metric < self.best_score + self.min_delta:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement. Counter {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement found
            self.best_score = metric
            self._save_model(model)
            self.counter = 0

    def _save_model(self, model):
        """
        Saves the model's state_dict if a save path is specified.

        Parameters:
            model (torch.nn.Module): The model to save.
        """
        if model and self.save_path:
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f"EarlyStopping: New best model saved to {self.save_path}")

    def reset(self):
        """
        Resets the early stopping parameters.
        """
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None
