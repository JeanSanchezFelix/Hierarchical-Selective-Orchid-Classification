import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import calculate_metrics, plot_metric_bar, plot_confusion_matrix, plot_radar_chart

def test_inference(model: nn.Module, test_loader: DataLoader, device: torch.device, save_dir: str):
    """
    Perform inference on the test dataset.

    Parameters:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to perform inference on (CPU or GPU).
        save_dir (str): Directory to save plots. If None, plots will not be saved.
    """
    logging.info("Starting inference on the test dataset...")
    
    model.eval()  # Set the model to evaluation mode
    predictions, ground_truths, probabilities = [], [], []
    class_labels = test_loader.dataset.classes

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in tqdm(test_loader, desc="Inference Progress"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    logging.info("Inference complete.")
    
    # Calculate metrics
    y_true = np.array(ground_truths)
    y_pred = np.array(predictions)
    y_proba = np.array(probabilities) if probabilities else None

    metrics = calculate_metrics(y_true, y_pred, y_proba)

    # Generate and save plots if `save_dir` is specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_metric_bar(metrics, save_path=os.path.join(save_dir, "metrics_bar_chart.png"))
        plot_confusion_matrix(
            metrics["Confusion Matrix"],
            labels=class_labels,
            save_path=os.path.join(save_dir, "confusion_matrix.png")
        )
        plot_radar_chart(
            {k: v for k, v in metrics.items() if k != "Confusion Matrix"},
            save_path=os.path.join(save_dir, "radar_chart.png")
        )
        logging.info(f"Plots saved to {save_dir}")