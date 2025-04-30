import os
import time
import logging
from typing import Callable, Optional, Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import KFold

from model_compression.src.utils.callbacks import Callback
from model_compression.src.utils.metrics import plot_train_val_curve, calculate_metrics
from model_compression.src.utils.model_setup import tf_setup
from model_compression.src.eval import test_inference, _compute_loss_and_predictions

def transfer_learning(
    model_name: str,
    data_loaders: Dict[str, DataLoader],
    save_dir: str,
    learning_rate: float = 0.001,
    num_epochs: int = 5,
    criterion_name: str = "cross_entropy",
    optimizer_name: str = "adam",
    callbacks: Optional[List[Callback]] = None,
    pretrained_weights_path: Optional[str] = None,
    use_class_weights: bool = False,
    freeze_base_layers: bool = True,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> nn.Module:
    """
    Train a model using transfer learning.

    Args:
        model_name: Name of the pre-trained model (e.g., 'resnet18').
        data_loaders: Dict with 'train', 'val', and optionally 'test' DataLoaders.
        save_dir: Directory to save checkpoints and metrics.
        learning_rate: Learning rate for optimizer.
        num_epochs: Number of training epochs.
        criterion_name: Loss function identifier ('cross_entropy', 'bce', etc.).
        optimizer_name: Optimizer identifier ('adam', 'sgd', etc.).
        callbacks: List of Callback instances to trigger during training.
        pretrained_weights_path: Path to pretrained weights file for fine-tuning.
        use_class_weights: Whether to compute and apply class weights.
        freeze_base_layers: Whether to freeze all layers except classifier.
        device: Computation device.

    Returns:
        model: The trained PyTorch model.

    Raises:
        RuntimeError: If training fails or model cannot be saved.
    """
    # Ensure save directories exist
    os.makedirs(save_dir, exist_ok=True)
    metrics_dir = os.path.join(save_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Extract loaders
    logging.info("Loading datasets...")
    train_loader = data_loaders.get("train")
    val_loader = data_loaders.get("val")
    if train_loader is None or val_loader is None:
        raise ValueError("Both 'train' and 'val' DataLoaders must be provided.")
    logging.info("Datasets loaded successfully.")
    
    # Set up model, criterion, optimizer
    try:
        model, criterion, optimizer = tf_setup(
            model_name=model_name,
            pretrained_weights_path=pretrained_weights_path,
            data_loader=train_loader,
            criterion_name=criterion_name,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            use_class_weights=use_class_weights,
        )
    except Exception as error:
        logging.error(f"Model setup failed: {error}")
        raise RuntimeError("Failed to setup model.")
    logging.info("Model setup complete.")

    # Optionally freeze base layers
    if freeze_base_layers:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze classifier layer
        if hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        logging.info("Base layers frozen; classifier unfrozen.")

    # Move model to device
    model.to(device)
    logging.info(f"Model moved to device: {device}")

    # Notify callbacks of training start
    if callbacks:
        for callback in callbacks:
            callback.on_train_start(logs={})

    # Training and evaluation loop
    logging.info("Starting training...")
    start_time = time.time()
    history = _train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        callbacks=callbacks or [],
        device=device
    )
    elapsed = time.time() - start_time
    logging.info(f"Training completed in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")

    # Trigger training end callbacks.
    if callbacks:
        for callback in callbacks:
            callback.on_train_end(logs={"model": model, "optimizer": optimizer})

    # Load best model checkpoint
    checkpoint_path = os.path.join(save_dir, f"{model_name}_best_model.pth")
    try:
        model_data = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(model_data)
        model.eval()
        logging.info(f"Best model weights loaded from: {checkpoint_path}")
    except Exception as e:
        logging.warning(f"Could not load best model weights from {checkpoint_path}: {e}")

    # Plot loss curves
    try:
        plot_train_val_curve(history, save_path=os.path.join(metrics_dir, "loss_curve.png"))
    except Exception as plot_error:
        logging.warning(f"Failed to plot loss curves: {plot_error}")

    if "test" in data_loaders:
        try:
            test_inference(model, data_loaders['test'], device, save_dir=metrics_dir)
        except Exception as test_error:
            logging.warning(f"Test inference failed: {test_error}")

    return model

def _train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    callbacks: List[Callback],
    device: torch.device
) -> Dict[str, List[float]]:
    """
    Run training and evaluation for each epoch.

    Args:
        model Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        learning_rate: Learning rate (for logging).
        criterion: Loss function.
        optimizer: Optimizer instance.
        num_epochs: Number of training epochs.
        device: Device to train on.
        callbacks: List of training callbacks.

    Returns:
        history: Dictionary with lists of training and validation losses.
    """
    history = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        logging.info("-" * 10)

        logs = {"model": model, 
                "batch_size": train_loader.batch_size, 
                "learning_rate": learning_rate, 
                "optimizer": optimizer,
                "criterion": criterion}

        # Start epoch callbacks
        for callback in callbacks:
            callback.on_epoch_start(epoch, logs)

        # Training step
        train_loss = _train_one_epoch(model=model, 
                            train_loader=train_loader, 
                            criterion=criterion, 
                            optimizer=optimizer, 
                            epoch=epoch,
                            num_epochs=num_epochs, 
                            device=device, 
                            logs=logs)
        history['train'].append(train_loss)  

        # Validate
        val_metrics = test_inference(model, val_loader, device, criterion)
        val_loss = val_metrics.get('loss', 0.0)
        history['val'].append(val_loss)

        logging.info(f"Val Loss: {val_loss:.4f}")
        logging.info("Val Metrics: " + ", ".join([f"{key.lower()}: {value:.4f}" for key, value in val_metrics.items() if key != "loss"]))
        logs.update({"val_loss": val_loss, **{f"val_{key.lower()}": value for key, value in val_metrics.items() if key != "loss"}})

        # End epoch callbacks
        for callback in callbacks:
            callback.on_epoch_end(epoch, logs)

        # Check for early stopping
        if any(getattr(callback, "early_stop", False) for callback in callbacks):
            break

    return history

def _train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    num_epochs: int,
    device: torch.device,
    logs: Dict[str, Any]
) -> float:
    """
    Perform training for one epoch.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Device to train on.
        epoch: Current epoch index.
        num_epochs: Total number of epochs.
        logs: Logging and callback dictionary.

    Returns:
        train_loss: Average training loss for the epoch.
    """
    model.train()
    running_loss, total_samples = 0.0, 0
    predictions, ground_truths, probabilities = [], [], []

    with tqdm(total=len(train_loader), desc=f"Train Epoch {epoch + 1}/{num_epochs}", leave=False) as pbar:
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() # Zero gradients during training

            outputs = model(inputs)
            loss, probs, preds = _compute_loss_and_predictions(outputs, labels, criterion)
            loss.backward()
            optimizer.step()

            # Update running loss and sample count
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            # Accumulate predictions and ground truths for metrics
            ground_truths.extend(labels.cpu().detach().numpy())
            probabilities.extend(probs.cpu().detach().numpy())
            predictions.extend(preds.cpu().detach().numpy())
            
            # Update progress bar with average loss so far
            pbar.set_postfix(loss=f"{running_loss / total_samples:.4f}")
            pbar.update(1)

    # Compute aggregated metrics after epoch
    train_loss = running_loss / total_samples

    # Calculate metrics
    y_true, y_pred = np.array(ground_truths), np.array(predictions)
    y_proba = np.array(probabilities) if probabilities else None
    metrics = calculate_metrics(y_true, y_pred, y_proba)

    logging.info(f"Train Loss: {train_loss:.4f}")
    logging.info(f"Train Metrics: " + ", ".join([f"{key.lower()}: {value:.4g}" for key, value in metrics.items()]))
    logs.update({f"train_loss": train_loss, **{f"train_{key.lower()}": value for key, value in metrics.items()}})
    return train_loss

# TODO: Work in progress (finish)

def cross_validation(
    model_name: str,
    dataset,
    k_folds: int,
    save_dir: str,
    learning_rate: float = 0.001,
    epochs: int = 5,
    criterion: str = "cross_entropy",
    optimizer: str = "adam",
    pretrained_weights: str = None,
    freeze_base: bool = True
) -> None:
    """
    Perform k-fold cross-validation for a given model and dataset.

    Args:
        model_name (str): Name of the pre-trained model.
        dataset: The dataset to be split for cross-validation.
        k_folds (int): Number of folds for cross-validation.
        save_dir (str): Directory to save the models and results.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train for each fold.
        criterion (str): Loss function.
        optimizer (str): Optimizer type.
        pretrained_weights (str): Path to pre-trained weights, if any.
        freeze_base (bool): Whether to freeze base model layers during training.
    """
    logging.info(f"Starting {k_folds}-fold cross-validation...")

    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_results = []

    os.makedirs(save_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        logging.info(f"Processing fold {fold + 1}/{k_folds}")

        # Create data loaders for the current fold
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        data_loaders = {"train": train_loader, "val": val_loader}

        # Train model for the current fold
        model = train_models(
            model_name=model_name,
            data_loaders=data_loaders,
            save_dir=os.path.join(save_dir, f"fold_{fold + 1}"),
            learning_rate=learning_rate,
            epochs=epochs,
            criterion=criterion,
            optimizer=optimizer,
            pretrained_weights=pretrained_weights,
            freeze_base=freeze_base,
        )

        # Evaluate model performance on validation set
        fold_accuracy = evaluate_model(model, val_loader)
        fold_results.append(fold_accuracy)
        logging.info(f"Fold {fold + 1} accuracy: {fold_accuracy:.4f}")

    # Summarize results
    mean_accuracy = sum(fold_results) / k_folds
    logging.info(f"Cross-validation complete. Mean accuracy: {mean_accuracy:.4f}")


def evaluate_model(model, val_loader) -> float:
    """
    Evaluate the model on the validation set.

    Args:
        model: Trained model.
        val_loader: DataLoader for the validation set.

    Returns:
        float: Validation accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total
