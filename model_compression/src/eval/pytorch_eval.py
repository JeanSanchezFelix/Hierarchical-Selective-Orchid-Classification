import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datasets.registry import DATASET_REGISTRY

from model_compression.src.quantization.utils.inspect import is_quantized_model
from model_compression.src.utils.model_setup import _setup_model
from model_compression.src.utils.metrics import calculate_metrics, plot_metric_bar, plot_confusion_matrix, plot_roc_auc_curve, plot_radar_chart
from model_compression.src.eval.predictions import _compute_predictions

def test_inference(model: nn.Module, data_loader: DataLoader, device, criterion=None, save_dir=None) -> dict | None:
    """
    Performs inference on a dataset, computes metrics, and optionally saves plots.

    Args:
        model (nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to perform inference on (CPU or GPU).
        save_dir (str): Directory to save plots. If None, plots will not be saved.
        quant (bool): Flag to indicate if the model is quantized. Default is False.
    """
    logging.info("Starting inference...")
    
    # Set model to evaluation mode.
    if is_quantized_model(model):
        model = torch.ao.quantization.allow_exported_model_train_eval(model)  # restores eval/train to call move_exported_model_* under the hood
    
    model.eval()

    running_loss, total_samples = 0.0, 0
    predictions, ground_truths, probabilities = [], [], []
    class_labels = data_loader.dataset.classes

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in tqdm(data_loader, desc="Inference Progress"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels).item() if criterion else 0.0
            probs, preds = _compute_predictions(outputs)

            batch_size = inputs.size(0)
            running_loss += loss * batch_size
            total_samples += batch_size

            ground_truths.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    logging.info("Inference complete.")   

    # Calculate metrics.
    y_true, y_pred = np.array(ground_truths), np.array(predictions)
    y_proba = np.array(probabilities) if probabilities else None
    metrics = calculate_metrics(y_true, y_pred, y_proba)

    if criterion:
        metrics["loss"] = running_loss / total_samples

    # Generate and save plots if save_dir is provided.
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_metric_bar(metrics, save_path=os.path.join(save_dir, "metrics_bar_chart.png"))
        if class_labels is not None:
            plot_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                labels=class_labels,
                save_path=os.path.join(save_dir, "confusion_matrix.png")
            )
        if y_proba is not None:
            plot_roc_auc_curve(y_true, y_proba, save_path=os.path.join(save_dir, "roc_auc_curve.png"))
        plot_radar_chart(metrics, save_path=os.path.join(save_dir, "radar_chart.png"))
        logging.info(f"Plots saved to {save_dir}")
    
    return metrics

def evaluate(model_path: str, metadata_path: str, img_size: int, dataset: str, save_dir: str = None):
    """
    Perform inference on the test dataset and evaluate model performance.

    Args:
        model_path (str): Path to the saved model file.
        metadata_path (str): Path to the metadata of saved model file.
        dataset (str): The name of the dataset that will be used for evaluation.
        save_dir (str): Directory to save plots. If None, plots will not be saved.
    """
    # Set up device: use GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the saved model and its parameters
    model_data = torch.load(model_path, weights_only=True, map_location=device)
    metadata = torch.load(metadata_path)
    criterion = metadata["criterion"]
    batch_size = metadata["batch_size"]

    # Image normalization parameters (standard for ImageNet-pretrained models)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Define preprocessing transformations for the test dataset
    basic_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),        # Resize images to fixed size
        transforms.ToTensor(),                          # Convert images to PyTorch tensors
        transforms.Normalize(mean=mean, std=std)        # Normalize using ImageNet stats
    ])

    # Load the test dataset using a registry (assuming DATASET_REGISTRY exists)
    test_dataset = DATASET_REGISTRY[dataset](mode='test', transform=basic_transforms)
    test_loader = DataLoader(test_dataset.dataset, batch_size=batch_size, shuffle=False)

    # Retrieve class labels from the dataset
    class_labels = test_loader.dataset.classes
    
    filename = model_path.split('/')[-1]                # Split the path by '/' and get the last component (filename)
    model_name = filename.split('_best_model.pth')[0]   # Split the filename by '_best_model.pth' to extract the model name

    # Initialize and load the model
    model = setup_model(model_name, pretrained_weights=False, num_classes=len(class_labels))
    model.load_state_dict(model_data)
    model.to(device)  # Move the model to the same device as the inputs

    test_inference(model, test_loader, device, criterion, save_dir)
