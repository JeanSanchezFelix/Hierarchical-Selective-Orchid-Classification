import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

def calculate_model_weights(dataloader: DataLoader) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets dynamically.

    Args:
        dataloader (DataLoader): PyTorch DataLoader containing the dataset.

    Returns:
        torch.Tensor: Class weights tensor for weighted loss computation.
    """
    # Get class labels from dataset
    class_labels = dataloader.dataset.classes
    num_classes = len(class_labels)

    # Extract targets and convert to numpy array
    targets = np.array(dataloader.dataset.targets)

    # Compute class weights using sklearn
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=targets)

    # Convert to PyTorch tensor
    return torch.tensor(class_weights, dtype=torch.float32)

def get_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to oversample minority classes.

    Args:
        dataset (dataset): PyTorch dataset.

    Returns:
        WeightedRandomSampler: Sampler object for DataLoader.
    """
    # Compute class counts dynamically
    targets = np.array(dataset.dataset.targets)
    class_counts = Counter(targets)
    
    # Compute sample weights (inverse of class frequency)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = np.array([class_weights[label] for label in targets])

    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
