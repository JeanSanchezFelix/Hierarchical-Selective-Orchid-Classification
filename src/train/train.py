import os
import time
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import KFold
from src.utils.metrics import plot_train_val_curve, calculate_metrics
from src.utils.model_setup import tf_setup
from src.utils.eval import test_inference, compute_loss_and_predictions

def transfer_learning(
    model_name: str,
    data_loaders: dict[str, DataLoader],
    save_dir: str,
    learning_rate: float = 0.001,
    epochs: int = 5,
    criterion: str = "cross_entropy",
    optimizer: str = "adam",
    callbacks: list = None,
    pretrained_weights: str = None,
    class_weights: bool = False,
    freeze_base: bool = True
):
    """
    Train a model using transfer learning 

    Args:
        model_name (str): Name of the pre-trained model to use (e.g., 'resnet18', 'mobilenet_v2').
        data_loaders dict[str, DataLoader]: DataLoaders containing data splits (e.g. train, val, test)
        save_dir (str): Directory to save the trained model and checkpoints.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        criterion (str): Criterion to use (e.g., 'cross_entropy', 'bce').
        optimizer (str): Optimizer to use (e.g., 'adam', 'sgd').
        callbacks (list): List of callbacks (e.g., 'ModelCheckpoint', 'EarlyStopping')
        pretrained_weights (str): Path to existing weights for further training. Defaults to None.
        class_weights (bool): Whether to compute class weights for imbalanced datasets.
        freeze_base (bool): Whether to freeze the base model layers.
    """
    # Load dataset
    logging.info("Loading datasets...")
    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]
    logging.info("Datasets loaded successfully.")
    
    model, criterion, optimizer = tf_setup(model_name, 
                                           learning_rate, 
                                           criterion, 
                                           optimizer, 
                                           pretrained_weights, 
                                           train_loader, 
                                           class_weights=class_weights)

    # Freeze base layers if specified
    if freeze_base:
        logging.info("Freezing base layers of the model.")
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "fc"):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logging.info(f"Model moved to device: {device}")

    # Initialize variables for checkpointing
    best_model_path = os.path.join(save_dir, f"{model_name}_best_model.pth")
    metrics_save_dir = os.path.join(save_dir, "metrics")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(metrics_save_dir, exist_ok=True)
    logging.info(f"Training will save checkpoints to: {save_dir}")

    for callback in callbacks:
        callback.on_train_start(logs={})

    # Training loop
    logging.info("Starting training...")
    start = time.time()

    loss_dict = _train_and_evaluate(model, train_loader, val_loader, learning_rate, criterion, optimizer, epochs, device, callbacks)

    time_elapsed = time.time() - start
    logging.info(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    # Trigger training end callbacks
    for callback in callbacks:
        callback.on_train_end(logs={"model": model, "optimizer": optimizer})

    # Load the best model weights before returning
    model_data = torch.load(best_model_path, weights_only=True)
    model.load_state_dict(model_data)
    logging.info(f"Best model weights loaded from: {best_model_path}")
    plot_train_val_curve(loss_dict, save_path=os.path.join(metrics_save_dir, "loss_curve.png"))

    if data_loaders["test"]:
        test_inference(model, data_loaders["test"], device, save_dir=metrics_save_dir)

    return model

def _train(model: nn.Module,
    train_loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    logs: dict
) -> float:
    model.train()
    running_loss, total_samples = 0.0, 0
    predictions, ground_truths, probabilities = [], [], []

    with tqdm(total=len(train_loader), desc=f"Train Epoch {epoch + 1}/{epochs}") as pbar:
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() # Zero gradients during training

            outputs = model(inputs)
            loss, probs, preds = compute_loss_and_predictions(outputs, labels, criterion)
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

def _train_and_evaluate(model, train_loader, val_loader, learning_rate, criterion, optimizer, epochs, device, callbacks) -> dict[str,list]:
    """
    Train and evaluate the model while displaying metrics with a progress bar.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        learning_rate (float): Learning rate used for logging
        criterion: Loss function.
        optimizer: Optimization algorithm.
        epochs (int): Number of epochs.
        device (torch.device): Device to run the model on.
    Returns:
        dict: Losses for each phase across all epochs.
    """
    loss_dict = {'train': [], 'val': []}

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        logging.info("-" * 10)

        logs = {"model": model, 
                "batch_size": train_loader.batch_size, 
                "learning_rate": learning_rate, 
                "optimizer": optimizer,
                "criterion": criterion,
                "epoch": epoch}

        # Trigger on_epoch_start callbacks
        for callback in callbacks:
            callback.on_epoch_start(epoch, logs)

        # Training step
        train_loss = _train(model, train_loader, criterion, optimizer, device, epoch, epochs, logs)
        loss_dict['train'].append(train_loss)  

        # Validation step using test_inference
        val_metrics = test_inference(model, val_loader, device, criterion=criterion, save_dir=None)
        val_loss = val_metrics.get("loss", 0.0)
        loss_dict['val'].append(val_loss)

        logging.info(f"Val Loss: {val_loss:.4f}")
        logging.info("Val Metrics: " + ", ".join([f"{key.lower()}: {value:.4f}" for key, value in val_metrics.items() if key != "loss"]))
        logs.update({"val_loss": val_loss, **{f"val_{key.lower()}": value for key, value in val_metrics.items() if key != "loss"}})

        # Trigger on_epoch_end callbacks
        for callback in callbacks:
            callback.on_epoch_end(epoch, logs)

        # Check for early stopping
        if any(getattr(cb, "early_stop", False) for cb in callbacks):
            break

    return loss_dict

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


# if __name__ == "__main__":
