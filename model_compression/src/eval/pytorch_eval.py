import os
import logging
from typing import Optional, Dict, Any, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets.registry import DATASET_REGISTRY
from model_compression.src.quantization.utils.inspect import is_quantized_model
from model_compression.src.utils.model_setup import setup_model
from model_compression.src.utils.metrics import (
    calculate_metrics,
    plot_metric_bar,
    plot_confusion_matrix,
    plot_roc_auc_curve,
    plot_radar_chart,
    plot_calibration_curve,
)
from model_compression.src.eval.predictions import _compute_predictions

def test_inference(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    save_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Run inference on a dataset, compute metrics, and optionally save visualizations.

    Args:
        model: Trained PyTorch model.
        data_loader: DataLoader for test or validation data.
        device: Device for inference (e.g., 'cpu' or 'cuda').
        criterion: Optional loss function to compute average loss.
        save_dir: Directory to save plots (bar chart, confusion matrix, ROC, calibration).

    Returns:
        Dictionary of computed metrics (including 'loss' if criterion is provided).

    Raises:
        RuntimeError: If dataset does not expose 'classes' attribute.
    """
    logging.info("Starting inference...")
    
    # Set model to evaluation mode.
    if is_quantized_model(model):
        # restores eval/train to call move_exported_model_* under the hood
        model = torch.ao.quantization.allow_exported_model_train_eval(model)

    model.to(device).eval()

    total_loss, total_samples = 0.0, 0
    y_true: List[int] = []
    y_pred: List[int] = []
    y_proba: List[List[float]] = []
    
    # Retrieve class labels
    classes = getattr(data_loader.dataset, 'classes', None)
    if classes is None:
        raise RuntimeError("Dataset must define a 'classes' attribute.")

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in tqdm(data_loader, desc="Inference Progress"):
            
            if len(classes) <= 2:   # Binary classification label adjustment
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            else:
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            batch_size = inputs.size(0)
            total_samples += batch_size

            # Compute loss
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * batch_size
                
            probs, preds = _compute_predictions(outputs)

            y_true.extend(labels.cpu().detach().numpy())
            y_pred.extend(preds.cpu().detach().numpy())
            y_proba.extend(probs.cpu().detach().numpy())

    logging.info("Inference complete.")   

    # Calculate metrics.
    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
    y_proba_arr = np.array(y_proba) if y_proba else None
    metrics = calculate_metrics(y_true_arr, y_pred_arr, y_proba_arr)

    if criterion:
        metrics["loss"] = total_loss / total_samples

    # Save plots if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_metric_bar(metrics, title="Test Metrics", save_path=os.path.join(save_dir, "metrics.png"))
        plot_confusion_matrix(
            y_true_arr, y_pred_arr, labels=classes,
            title="Confusion Matrix", save_path=os.path.join(save_dir, "confusion_matrix.png")
        )
        plot_radar_chart(metrics, save_path=os.path.join(save_dir, "radar_chart.png"))
        if y_proba_arr is not None:
            classes = data_loader.dataset.classes
            plot_roc_auc_curve(
                y_true_arr, y_proba_arr, classes,
                title="ROC-AUC Curve", save_path=os.path.join(save_dir, "roc_auc.png")
            )
            plot_calibration_curve(
                y_true_arr, y_proba_arr, classes,
                title="Calibration Curve", save_path=os.path.join(save_dir, "calibration.png")
            )
        logging.info(f"Saved evaluation plots to {save_dir}")
    
    return metrics

def evaluate(
    model_path: str,
    metadata_path: str,
    img_size: int,
    dataset_name: str,
    save_dir: Optional[str] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    """
    Load a saved model and metadata, run evaluation on the test set, and save results.

    Args:
        model_path: Path to saved model state dict.
        metadata_path: Path to metadata file (dict with 'criterion', 'batch_size').
        img_size: Image dimension for resizing input.
        dataset_name: Key into DATASET_REGISTRY to load data.
        save_dir: Optional directory to save plots.

    Raises:
        FileNotFoundError: If model or metadata files are missing.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load metadata
    metadata = torch.load(metadata_path, map_location=device, weights_only=False)
    criterion = metadata.get('criterion')
    batch_size = metadata.get('batch_size', 32)

    # Image normalization parameters (standard for ImageNet-pretrained models)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Define preprocessing transformations for the test dataset
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),        # Resize images to fixed size
        transforms.ToTensor(),                          # Convert images to PyTorch tensors
        transforms.Normalize(mean=mean, std=std)        # Normalize using ImageNet stats
    ])

    # Load the test dataset using a registry
    if dataset_name not in DATASET_REGISTRY:
        raise KeyError(f"Dataset '{dataset_name}' not found in registry.")
    ds_cls = DATASET_REGISTRY[dataset_name]
    test_ds = ds_cls(mode='test', transform=transform)
    test_loader = DataLoader(test_ds.dataset, batch_size=batch_size, shuffle=False)

    # Extract class labels
    classes = getattr(test_loader.dataset, 'classes', None)

    # Infer number of classes
    num_classes = len(classes)
    
    # Instantiate model architecture and load weights
    base_name = os.path.basename(model_path).split('.')[0][:-len("_best_model")]
    model = setup_model(base_name, None, num_classes)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    metrics = test_inference(model, test_loader, device, save_dir=save_dir)
    logging.info(f"Final evaluation metrics: {metrics}")
