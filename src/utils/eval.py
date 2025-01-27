import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datasets.registry import DATASET_REGISTRY
from src.utils.model_setup import setup_model
from src.utils.metrics import calculate_metrics, plot_metric_bar, plot_confusion_matrix, plot_radar_chart

import os
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

def evaluate(model_path: str, img_size: int, dataset: str, save_dir: str = None):
    """
    Perform inference on the test dataset and evaluate model performance.

    Parameters:
        model_path (str): Path to the saved model file.
        dataset (str): The name of the dataset that will be used for evaluation.
        save_dir (str): Directory to save plots. If None, plots will not be saved.
    """
    # Set up device: use GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the saved model and its parameters
    model_data = torch.load(model_path, weights_only=False, map_location=device)
    batch_size = model_data["batch_size"]

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

    # Initialize and load the model
    model = setup_model("mobilenet_v2", pretrained_weights=False, num_classes=len(class_labels))
    model.load_state_dict(model_data["model"])
    model.to(device)  # Move the model to the same device as the inputs

    test_inference(model, test_loader, device, save_dir)


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

            ground_truths.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
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
            y_true=y_true,
            y_pred=y_pred,
            labels=class_labels,
            save_path=os.path.join(save_dir, "confusion_matrix.png")
        )
        plot_radar_chart(metrics, save_path=os.path.join(save_dir, "radar_chart.png"))
        logging.info(f"Plots saved to {save_dir}")