import os
import time
import logging
import copy
from typing import Callable, Optional, Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_compression.src.utils.callbacks import Callback
from model_compression.src.utils.metrics import calculate_metrics, plot_train_val_curve
from model_compression.src.eval import _compute_loss_and_predictions, test_inference
from model_compression.src.quantization.core.quantize import quantize_pytorch_model
from model_compression.src.quantization.utils.model_setup import kd_setup

def train_kd(
    teacher_name: str,
    student_name: str,
    data_loaders: Dict[str, DataLoader],
    save_dir: str,
    learning_rate: float = 0.001,
    num_epochs: int = 5,
    criterion_name: str = "cross_entropy",
    optimizer_name: str = "adam",
    callbacks: Optional[List[Callback]] = None,
    teacher_weights_path: Optional[str] = None,
    temperature: float = 2.0,
    distill_loss_weight: float = 0.25,
    ce_loss_weight: float = 0.75,
    quant_mode: Optional[str] = None,
    qat_config: Optional[str] = None,
    use_class_weights: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Tuple[nn.Module, nn.Module]:
    """
    Train a student model using Knowledge Distillation (KD) with optional Quantization-Aware Training (QAT).
    
    This function performs the following steps:
      1. Loads the teacher and student models using their respective model names. The student model is loaded
         as a standard model and then prepared for QAT via quantization_mode.
      2. Configures the loss function and optimizer using helper functions.
      3. Runs the training loop for a specified number of epochs while applying knowledge distillation 
         (combining soft targets from the teacher with the true label loss).
      4. During training, checkpoints and callbacks are handled. After training, the best model weights 
         are loaded and the student model is quantized for export if quant_mode is provided.
      5. If a test DataLoader is provided, the student model is evaluated.
    
    Args:
        teacher_name: Name of the teacher model architecture.
        student_name: Name of the student model architecture.
        data_loaders: Dictionary of DataLoaders for 'train', 'val', and optional 'test'.
        save_dir: Directory to save checkpoints and metrics.
        learning_rate: Learning rate for optimizer.
        num_epochs: Number of training epochs.
        loss_name: Identifier for the loss function.
        optimizer_name: Identifier for the optimizer.
        callbacks: List of Callback instances for training events.
        teacher_weights_path: Path to pretrained teacher weights.
        temperature: Softening temperature for KD.
        distill_loss_weight: Weight for soft target loss.
        ce_loss_weight: Weight for cross-entropy loss.
        quant_mode: QAT mode ('eager', 'fx', 'export').
        qat_config: QAT configuration name.
        use_class_weights: Whether to apply class weights.
        device: Computation device.
    
    Returns:
        A tuple containing
            - teacher: The loaded teacher model.
            - student: The student model after training.

    Raises:
        ValueError: If 'train' or 'val' loaders are missing.
        RuntimeError: If model setup fails.
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
        teacher_model, student_model, criterion, optimizer = kd_setup(
            teacher=teacher_name,
            student=student_name,
            learning_rate=learning_rate,
            criterion_name=criterion_name,
            optimizer_name=optimizer_name,
            data_loader=train_loader,
            class_weights=use_class_weights,
            teacher_model_weights=teacher_weights_path,
            quant_mode=quant_mode,
            config=qat_config,
            device=device
        )
    except Exception as e:
        logging.error(f"KD setup failed: {e}")
        raise RuntimeError("Failed to setup KD models.")
    logging.info("Model setup complete.")

    teacher_model.eval()
    teacher_model.to(device)
    student_model.to(device)
    logging.info("Models loaded and moved to device.")

    # Prepare checkpoint path
    ext = f"_qat_kd" if quant_mode else "_kd"
    checkpoint_path = os.path.join(save_dir, f"{student_name}{ext}.pth")
    logging.info(f"Checkpoints and metrics will be saved to: {checkpoint_path}")

    # Notify callbacks of training start
    if callbacks:
        for callback in callbacks:
            callback.on_train_start(logs={})

    # Training loop
    logging.info("Starting KD training...")
    start_time = time.time()
    history = _train_and_evaluate_kd(
        teacher=teacher_model,
        student=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        temperature=temperature,
        distill_loss_weight=distill_loss_weight,
        ce_loss_weight=ce_loss_weight,
        quant_mode=quant_mode,
        callbacks=callbacks or [],
        device=device
    )
    elapsed = time.time() - start_time
    logging.info(f"KD training completed in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")


    # Trigger training end callbacks.
    if callbacks:
        for callback in callbacks:
            callback.on_train_end(logs={"model": student_model, "optimizer": optimizer})

    # Load the best student model weights from checkpoint.
    try:
        model_data = torch.load(checkpoint_path, weights_only=True)
        student_model.load_state_dict(model_data)
        logging.info(f"Loaded best student model weights from {checkpoint_path}")
    except Exception as e:
        logging.warning(f"Could not load best model weights from {checkpoint_path}: {e}")
        raise RuntimeError("Could not load best model weights from") from e

    # Optional quantization export
    if quant_mode:
        try:
            student_model = quantize_pytorch_model(
                model=student_model.cpu(),
                quant_mode=quant_mode,
                save_path=os.path.join(save_dir, "quantized_state.pth")
            )
            logging.info("Student model quantized for export.")
        except Exception as e:
            logging.warning(f"Quantization failed: {e}")
            raise RuntimeError("Quantization failed") from e

    # Plot training and validation loss curves.
    try:
        plot_train_val_curve(history, save_path=os.path.join(metrics_dir, "loss_curve.png"))
    except Exception as plot_error:
        logging.warning(f"Failed to plot loss curves: {plot_error}")

    # If a test set is provided, perform inference evaluation on the quantized student model.
    if "test" in data_loaders:
        try:
            test_inference(model=student_model, data_loader=data_loaders['test'], device=device, save_dir=metrics_dir)
        except Exception as test_error:
            logging.warning(f"Test inference failed: {test_error}")
            raise RuntimeError("Failed to test inference") from e

    return teacher_model, student_model


def _train_and_evaluate_kd(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    temperature: float,
    distill_loss_weight: float,
    ce_loss_weight: float,
    quant_mode: Optional[str],
    callbacks: List[Callback],
    device: torch.device
) -> Dict[str, List[float]]:
    """
    Run KD training and evaluation for each epoch.
    
    The total loss is a weighted sum of:
      - The distillation loss computed as KL divergence between softened teacher and student logits.
      - The standard loss computed on true labels.
      
    During validation, the student model is switched to evaluation mode (and optionally quantized) 
    to simulate the behavior of the final quantized model. Training and validation metrics are logged 
    and stored per epoch.
    
    Args:
        teacher: Teacher model in eval mode.
        student: Student model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        learning_rate: Learning rate for optimizer.
        criterion: Loss function.
        optimizer: Optimizer for student.
        num_epochs: Number of epochs to run.
        temperature: KD softening temperature.
        distill_loss_weight: Weight for distillation loss.
        ce_loss_weight: Weight for label loss.
        quant_mode: Quantization mode for QAT.
        callbacks: Callbacks for epoch events.
        device: Computation device
    
    Returns:
        Dictionary with lists of training and validation losses.
    """
    history = {'train': [], 'val': []}
    # Set teacher to evaluation mode (its parameters won't be updated)
    teacher.eval()
    teacher = teacher.to(device)

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        logging.info("-" * 10)

        logs = {"model": student, 
                "batch_size": train_loader.batch_size, 
                "learning_rate": learning_rate, 
                "optimizer": optimizer,
                "criterion": criterion}

        # Start epoch callbacks
        for callback in callbacks:
            callback.on_epoch_start(epoch, logs)

        # Training step
        train_loss = _train_one_epoch_kd(
            student=student,
            teacher=teacher,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            num_epochs=num_epochs,
            quant_mode=quant_mode,
            temperature=temperature,
            distill_loss_weight=distill_loss_weight,
            ce_loss_weight=ce_loss_weight,
            device=device,
            logs=logs,
        )
        history['train'].append(train_loss)

        # Validate 
        if quant_mode:
            try:
                student_copy = copy.deepcopy(student)
                quantized_for_val = quantize_pytorch_model(
                    model=student_copy.cpu(),
                    quant_mode=quant_mode
                )
                logging.info("Student model quantized for validation.")
            except Exception as e:
                logging.warning(f"Quantization failed for validation: {e}")
                raise RuntimeError("Quantization failed for validation") from e
            
        val_metrics = test_inference(quantized_for_val, val_loader, device, criterion=criterion, save_dir=None)
        val_loss = val_metrics.get("loss", 0.0)
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

def _train_one_epoch_kd(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    num_epochs: int,
    quant_mode: Optional[str],
    temperature: float,
    distill_loss_weight: float,
    ce_loss_weight: float,
    device: torch.device,
    logs: Dict[str, Any],
) -> float:
    """
    Perform one epoch of KD training, combining distillation and label losses.

    Args:
        student: Student model.
        teacher: Teacher model in eval mode.
        train_loader: Training DataLoader.
        loss_fn: Function yielding (label_loss, probabilities, predictions).
        optimizer: Optimizer for student.
        epoch: Current epoch index.
        num_epochs: Total number of epochs.
        quant_mode: QAT mode.
        temperature: KD softening temperature.
        distill_loss_weight: Weight for distillation loss.
        ce_loss_weight: Weight for label loss.
        device: Computation device.
        logs: Logging and callback dictionary.

    Returns:
        Average combined loss for the epoch.
    """
    if quant_mode:
        student = torch.ao.quantization.allow_exported_model_train_eval(student)

    student.train()
    
    total_loss, total_samples = 0.0, 0
    y_true: List[int] = []
    y_pred: List[int] = []
    y_proba: List[List[float]] = []

    with tqdm(total=len(train_loader), desc=f"Train Epoch {epoch + 1}/{num_epochs}", leave=True) as pbar:
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() # Zero gradients during training

            # Teacher forward pass without gradient computation.
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Student forward
            student_logits = student(inputs)

            # Compute soft targets loss using KL divergence.
            soft_loss = _compute_distillation_loss(teacher_logits=teacher_logits, student_logits=student_logits, T=temperature)

            # Compute the true label loss and obtain predictions/probabilities
            label_loss, probs, preds = _compute_loss_and_predictions(student_logits, labels, criterion)
            
            # Combine the losses using the specified weights.
            loss = distill_loss_weight * soft_loss + ce_loss_weight * label_loss
            loss.backward()
            optimizer.step()

            # Update running loss and sample count
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Accumulate predictions and ground truths for metrics
            y_true.extend(labels.cpu().detach().numpy())
            y_pred.extend(preds.cpu().detach().numpy())
            y_proba.extend(probs.cpu().detach().numpy())
            
            # Update progress bar with average loss so far
            pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")
            pbar.update(1)

    # Compute aggregated metrics after epoch
    train_loss = total_loss / total_samples

    # Calculate metrics
    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
    y_proba_arr = np.array(y_proba) if y_proba else None
    metrics = calculate_metrics(y_true_arr, y_pred_arr, y_proba_arr)

    logging.info(f"Train Loss: {train_loss:.4f}")
    logging.info(f"Train Metrics: " + ", ".join([f"{key.lower()}: {value:.4g}" for key, value in metrics.items()]))
    logs.update({f"train_loss": train_loss, **{f"train_{key.lower()}": value for key, value in metrics.items()}})
    return train_loss

def _compute_distillation_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    T: float
) -> torch.Tensor:
    """
    Computes the distillation loss using KL divergence with softened logits.
    
    The teacher and student logits are scaled by the temperature T, and the KL divergence 
    is computed with 'batchmean' reduction. The loss is scaled by T^2 as recommended in 
    distillation literature.
    
    Args:
        teacher_logits: Logits from teacher.
        student_logits: Logits from student.
        T: Softening temperature.
    
    Returns:
        KL-divergence loss scaled by T^2.
    """
    # Compute KL divergence loss between softened logits.
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T ** 2)
    return soft_loss
