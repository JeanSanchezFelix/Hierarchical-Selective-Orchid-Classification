import logging
from collections import Counter
from typing import Union

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, WeightedRandomSampler

### TODO: TEST IF THIS WORKS ###
def calculate_model_weights(
    dataset: Dataset
) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets to use in loss functions.

    Args:
        dataset: A PyTorch Dataset or Subset with attributes 'classes' and 'targets'.

    Returns:
        A 1D tensor of shape (num_classes,) containing the weight for each class.

    Raises:
        AttributeError: If dataset lacks 'classes' or 'targets'.
        ValueError: If 'targets' and 'classes' sizes mismatch.
    """
    # Access base dataset
    base = getattr(dataset, 'dataset', dataset)

    classes = getattr(base, 'classes', None)
    targets = getattr(base, 'targets', None)
    if classes is None or targets is None:
        logging.error("Dataset must have 'classes' and 'targets' attributes.")
        raise AttributeError("Dataset missing 'classes' or 'targets'.")

    num_classes = len(classes)
    targets_arr = np.array(targets)
    if targets_arr.ndim != 1:
        raise ValueError("`targets` must be a 1D array-like of class indices.")

    invalid_labels = np.setdiff1d(np.unique(targets_arr), np.arange(num_classes))
    if invalid_labels.size:
        raise ValueError(f"`targets` contains labels outside [0, {num_classes - 1}]: {invalid_labels.tolist()}")

    observed_classes = np.unique(targets_arr)
    if observed_classes.size == 0:
        raise ValueError("Cannot calculate class weights from an empty training split.")

    # A taxonomy can intentionally retain directories with zero images for
    # reporting. They do not occur in the training split and therefore cannot
    # receive a data-derived balancing weight.
    missing_classes = np.setdiff1d(np.arange(num_classes), observed_classes)
    if missing_classes.size:
        missing_labels = [classes[index] for index in missing_classes]
        logging.warning(
            "No training samples for %d declared class(es); assigning zero loss weight: %s",
            missing_classes.size,
            missing_labels,
        )

    observed_weights = compute_class_weight(
        class_weight='balanced',
        classes=observed_classes,
        y=targets_arr
    )
    weights = np.zeros(num_classes, dtype=np.float32)
    weights[observed_classes] = observed_weights
    return torch.tensor(weights, dtype=torch.float32)

def get_weighted_sampler(
    dataset: Dataset
) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to oversample underrepresented classes.

    Args:
        dataset: A PyTorch Dataset or Subset with attributes 'targets'.

    Returns:
        A WeightedRandomSampler for use in a DataLoader.

    Raises:
        AttributeError: If dataset lacks 'targets'.
    """
    base = getattr(dataset, 'dataset', dataset)
    targets = getattr(base, 'targets', None)
    if targets is None:
        logging.error("Dataset must have 'targets' attribute.")
        raise AttributeError("Dataset missing 'targets'.")

    labels = np.array(targets)
    class_counts = Counter(labels)
    # Compute inverse frequency for each class
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    # Assign weight to each sample
    sample_weights = np.array([class_weights[int(label)] for label in labels], dtype=np.float32)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
