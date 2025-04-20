import time
import logging
from tqdm import tqdm
from typing import Callable, Optional, Any
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import (load_data, process_callbacks, benchmark, compute_loss_and_predictions, calculate_metrics, 
                       plot_train_val_curve, test_inference)

from src.Quantization.quantization_utils.model_setup import qat_kd_setup
from src.Quantization.quantization_utils.conversions.litert import quantize_pytorch_model, convert_pytorch_model_to_tflite, check_quantized_modules

def compute_distillation_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor, T: float) -> torch.Tensor:
    """
    Computes the distillation loss using KL divergence with softened logits.
    
    The teacher and student logits are scaled by the temperature T, and the KL divergence 
    is computed with 'batchmean' reduction. The loss is scaled by T^2 as recommended in 
    distillation literature.
    
    Parameters:
        teacher_logits (torch.Tensor): Logits from the teacher model.
        student_logits (torch.Tensor): Logits from the student model.
        T (float): Temperature to soften the logits.
    
    Returns:
        torch.Tensor: The computed distillation loss.
    """
    # Compute KL divergence loss between softened logits.
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T ** 2)
    return soft_loss

def train_knowledge_distillation(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float,
    epochs: int,
    criterion: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    callbacks: list[Any],
    quant_mode: str,
    T: float,
    soft_target_loss_weight: float,
    ce_loss_weight: float,
    device: torch.device
) -> dict[str, list[float]]:
    """
    Trains the student model using knowledge distillation from the teacher model.
    
    The total loss is a weighted sum of:
      - The distillation loss computed as KL divergence between softened teacher and student logits.
      - The standard loss computed on true labels.
      
    During validation, the student model is switched to evaluation mode (and optionally quantized) 
    to simulate the behavior of the final quantized model. Training and validation metrics are logged 
    and stored per epoch.
    
    Parameters:
        teacher (nn.Module): The pre-trained teacher model.
        student (nn.Module): The student model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        learning_rate (float): The learning rate used (for logging purposes).
        epochs (int): Number of training epochs.
        criterion (Callable): Loss function for true label loss (e.g., nn.CrossEntropyLoss). 
                              This function should return a tuple: (label_loss, probabilities, predictions).
        optimizer (torch.optim.Optimizer): Optimizer for updating the student model parameters.
        callbacks (list[Any]): List of callback objects with methods on_epoch_start and on_epoch_end.
        quant_mode (str): Quantization mode string (e.g., 'eager', 'fx', or 'export') used during validation.
        T (float): Temperature for softening the logits.
        soft_target_loss_weight (float): Weight for the distillation (soft target) loss.
        ce_loss_weight (float): Weight for the cross entropy loss computed on true labels.
        device (torch.device): Device to perform training on.
    
    Returns:
        dict[str, list[float]]: A dictionary containing training and validation loss history 
                                  with keys 'train' and 'val'.
    """
    loss_dict = {'train': [], 'val': []}
    # Set teacher to evaluation mode (its parameters won't be updated)
    teacher.eval()
    
    for epoch in range(epochs):
        logs = {"model": student, 
                "batch_size": train_loader.batch_size, 
                "learning_rate": learning_rate, 
                "optimizer": optimizer,
                "criterion": criterion,
                "epoch": epoch}

        # Trigger on_epoch_start callbacks
        for callback in callbacks:
            callback.on_epoch_start(epoch, logs)

        for phase in ['train', 'val']:
            is_train = phase == 'train'

            # For training phase, we use the original student model.
            # For validation, we create a deep copy and quantize it.
            if is_train:
                # current_student = student.to(device)
                # Ensure the model is in train mode
                torch.ao.quantization.move_exported_model_to_train(student)
            else:
                # Create a deep copy to avoid altering the training model
                # current_student = copy.deepcopy(student).to(device)
                # Quantize the copied student model for validation
                # Pass save_dir as None to avoid saving the state dict
                # current_student = quantize_pytorch_model(current_student, quant_mode=quant_mode, save_dir=None)
                # Switch to evaluation mode to perform inference
                torch.ao.quantization.move_exported_model_to_eval(student)

            data_loader = train_loader if is_train else val_loader

            running_loss = 0.0
            total_samples  = 0.0
            predictions, ground_truths, probabilities = [], [], []

            with tqdm(total=len(data_loader), desc=f"{phase.capitalize()} Epoch {epoch + 1}/{epochs}") as pbar:
                for inputs, labels in data_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Zero gradients only during training
                    if is_train:
                        optimizer.zero_grad()
                    
                    # Teacher forward pass without gradient computation.
                    with torch.no_grad():
                        teacher = teacher.to(device)
                        teacher_logits = teacher(inputs)
                    
                    # Enable gradients only in training phase
                    with torch.set_grad_enabled(is_train):                        
                        student = student.to(device)
                        student_logits = student(inputs)

                        # Compute soft targets loss using KL divergence.
                        soft_loss = compute_distillation_loss(teacher_logits=teacher_logits, student_logits=student_logits, T=T)

                        # Compute the true label loss and obtain predictions/probabilities
                        label_loss, probs, preds = compute_loss_and_predictions(student_logits, labels, criterion)
                        
                        # Combine the losses using the specified weights.
                        loss = soft_target_loss_weight * soft_loss + ce_loss_weight * label_loss

                        # Backpropagate and update weights if in training mode.
                        if is_train:
                            loss.backward()
                            optimizer.step()
                    
                    # Update running loss and sample count
                    batch_size = inputs.size(0)
                    running_loss += loss.item() * batch_size
                    total_samples += batch_size

                    # Accumulate predictions and ground truths for metrics
                    ground_truths.extend(labels.cpu().detach().numpy())
                    predictions.extend(preds.cpu().detach().numpy())
                    probabilities.extend(probs.cpu().detach().numpy())
                    
                    # Update progress bar with average loss so far
                    pbar.set_postfix(loss=f"{running_loss / total_samples:.4f}")
                    pbar.update(1)
                
            # Compute aggregated metrics after epoch
            epoch_loss = running_loss / total_samples
            
            y_true = np.array(ground_truths)
            y_pred = np.array(predictions)
            y_proba = np.array(probabilities) if probabilities else None

            # Calculate metrics
            metrics = calculate_metrics(y_true, y_pred, y_proba)

            loss_dict[phase].append(epoch_loss)
            logging.info(f"{phase.capitalize()} Loss: {epoch_loss:.4f}")
            logging.info(f"{phase.capitalize()} Metrics: " + ", ".join([f"{key}: {value:.4g}" for key, value in metrics.items()]))
            logs.update({f"{phase}_loss": epoch_loss, **metrics})

        # Trigger on_epoch_end callbacks
        for callback in callbacks:
            callback.on_epoch_end(epoch, logs)

        # Check for early stopping
        if any(getattr(cb, "early_stop", False) for cb in callbacks):
            break

    return loss_dict

def train_qat_kd(
    teacher_name: str,
    student_name: str,
    data_loaders: dict[str, DataLoader],
    save_dir: str,
    learning_rate: float = 0.001,
    epochs: int = 5,
    criterion: str = "cross_entropy",
    optimizer: str = "adam",
    callbacks: Optional[list[Any]] = None,
    teacher_model_weights: Optional[str] = None,
    quant_mode: str = "export",
    config: str = "qnnpack",
    class_weights: bool = False,
    device: torch.device = torch.device("cuda")
) -> tuple[nn.Module, nn.Module]:
    """
    Trains a student model using Knowledge Distillation (KD) combined with Quantization-Aware Training (QAT).
    
    This function performs the following steps:
      1. Loads the teacher and student models using their respective model names. The student model is loaded
         as a standard model and then prepared for QAT via quantization_mode.
      2. Configures the loss function and optimizer using helper functions.
      3. Runs the training loop for a specified number of epochs while applying knowledge distillation 
         (combining soft targets from the teacher with the true label loss).
      4. During training, checkpoints and callbacks are handled. After training, the best model weights 
         are loaded and the student model is quantized for export.
      5. If a test DataLoader is provided, the quantized student model is evaluated.
    
    Parameters:
        teacher_name (str): Name of the pre-trained teacher model (e.g., 'resnet50', 'efficientnet_b0').
        student_name (str): Name of the student model to be trained with QAT.
        data_loaders (dict[str, DataLoader]): Dictionary containing DataLoaders for dataset splits 
                                              (keys: 'train', 'val', and optionally 'test').
        save_dir (str): Directory where the trained model, checkpoints, and metrics will be saved.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        epochs (int, optional): Number of training epochs. Defaults to 5.
        criterion (str, optional): Loss function name (e.g., 'cross_entropy', 'bce'). Defaults to 'cross_entropy'.
        optimizer (str, optional): Optimizer name (e.g., 'adam', 'sgd'). Defaults to 'adam'.
        callbacks (Optional[list[Any]], optional): List of callback objects for monitoring, checkpointing,
                                                     and early stopping. Defaults to None.
        teacher_model_weights (Optional[str], optional): Path to pre-trained weights for the teacher model.
                                                         Defaults to None.
        quant_mode (str, optional): Quantization mode to use for preparing the student model 
                                    ('eager', 'fx', or 'export'). Defaults to 'export'.
        config (str, optional): QAT configuration identifier ('qnnpack' for default QAT config for qnnpack 
                                or custom). Defaults to 'qnnpack'.
        class_weights (bool, optional): Whether to compute and apply class weights for imbalanced datasets.
                                        Defaults to False.
        device (torch.device, optional): Device to perform operations on. Defaults to torch.device("cuda").
    
    Returns:
        tuple[nn.Module, nn.Module]: A tuple containing:
            - teacher_model: The loaded teacher model.
            - quantized_student: The quantized student model after training.
    
    Notes:
        - The student model is prepared for QAT using example inputs from the training DataLoader.
        - After training, the best model weights are loaded from a checkpoint, and the student model is quantized 
          (using quantize_pytorch_model) for export.
        - Training metrics (loss curves and evaluation metrics) are saved to the specified directory.
    """
    # Load dataset
    logging.info("Loading datasets...")
    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]
    logging.info("Datasets loaded successfully.")
    
    # Set up teacher and student models, loss function, and optimizer.
    teacher, student, criterion_fn, optimizer_obj = qat_kd_setup(teacher=teacher_name, 
                                                                student=student_name,
                                                                learning_rate=learning_rate, 
                                                                criterion=criterion, 
                                                                optimizer=optimizer, 
                                                                teacher_model_weights=teacher_model_weights, 
                                                                dataloader=train_loader, 
                                                                quant_mode=quant_mode,
                                                                config=config,
                                                                class_weights=class_weights,
                                                                device=torch.device("cpu"))
    logging.info("Model setup complete.")

    # Set up checkpoint and metrics directories.
    quantized_model_path = os.path.join(save_dir, f"{student_name}_qat_kd.pth")
    metrics_save_dir = os.path.join(save_dir, "metrics")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(metrics_save_dir, exist_ok=True)
    logging.info(f"Checkpoints and metrics will be saved to: {save_dir}")

    # Trigger training start callbacks.
    if callbacks is not None:
        for callback in callbacks:
            callback.on_train_start(logs={})

    # Training loop
    logging.info("Starting training...")
    start_time = time.time()

    # Train the student model using knowledge distillation.
    loss_dict = train_knowledge_distillation(
                            teacher=teacher, 
                            student=student, 
                            train_loader=train_loader,
                            val_loader=val_loader,
                            learning_rate=learning_rate,
                            epochs=epochs, 
                            criterion=criterion_fn,
                            optimizer=optimizer_obj,
                            callbacks=callbacks,
                            quant_mode=quant_mode,
                            T=2, 
                            soft_target_loss_weight=0.25, 
                            ce_loss_weight=0.75, 
                            device=device
                        )
    elapsed_time = time.time() - start_time
    logging.info(f"Training complete in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")

    # Trigger training end callbacks.
    if callbacks is not None:
        for callback in callbacks:
            callback.on_train_end(logs={"model": student, "optimizer": optimizer_obj})

    # Load the best student model weights from checkpoint.
    model_data = torch.load(quantized_model_path, weights_only=True)
    student.load_state_dict(model_data)
    logging.info(f"Best model weights loaded from: {quantized_model_path}")


    # Quantize the student model for export.
    quantized_student = quantize_pytorch_model(student.to("cpu"), quant_mode, save_dir=os.path.join(save_dir, "quantized_state.pth"))
    # Plot training and validation loss curves.
    plot_train_val_curve(loss_dict, save_path=os.path.join(metrics_save_dir, "loss_curve.png"))

    # If a test set is provided, perform inference evaluation on the quantized student model.
    if "test" in data_loaders and data_loaders["test"] is not None:
        test_inference(quantized_student, data_loaders["test"], torch.device("cpu"), metrics_save_dir)

    return teacher, quantized_student

