import os
import logging
from collections import Counter
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Sampler
from torchvision import transforms

from datasets.registry import DATASET_REGISTRY
from model_compression.src.utils.data_imbalance import get_weighted_sampler

def _log_dataset_statistics(
    dataset: Dataset,
    dataset_label: str
) -> None:
    """
    Log size and per-class distribution of a dataset.

    Args:
        dataset: A Dataset or a Subset wrapping a dataset.
        dataset_label: Label for logging (e.g., 'train', 'val', 'test').
    """
    # Determine underlying dataset and indices
    if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        base = dataset.dataset
        indices = dataset.indices
    else:
        base = dataset
        indices = list(range(len(dataset)))

    # Extract class names and targets
    classes = getattr(base, 'classes', None)
    targets = getattr(base, 'targets', None)
    if targets is None:
        logging.warning(f"Dataset {dataset_label} has no 'targets' attribute.")
        return

    # Compute counts
    selected_targets = [targets[i] for i in indices]
    counts = Counter(selected_targets)

    total = len(indices)
    logging.info(f"{dataset_label.capitalize()} size: {total}")
    for class_idx, count in sorted(counts.items()):
        name = classes[class_idx] if classes else str(class_idx)
        logging.info(f"  Class '{name}': {count} samples")


def _log_all_statistics(
    loaders: Dict[str, DataLoader]
) -> None:
    """
    Log statistics for all provided DataLoaders.

    Args:
        Mapping of split names to DataLoader instances.
    """
    total = 0
    for label, loader in loaders.items():
        dataset = loader.dataset
        _log_dataset_statistics(dataset, label)
        total += len(dataset)
    logging.info(f"Total samples across all splits: {total}")

def load_data(
    dataset_name: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    test_split: float = 0.1,
    img_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    use_augmentation: bool = False,
    use_sampler: bool = False
) -> Dict[str, DataLoader]:
    """
    Load a dataset by name, apply transforms, split if needed, and return DataLoaders.

    Args:
        dataset_name: Key for DATASET_REGISTRY to load.
        batch_size: Batch size for DataLoaders.
        train_split: Fraction for training split when manual splitting.
        test_split: Fraction for test split when manual splitting.
        img_size: Image resize size.
        mean: Normalization means.
        std: Normalization stds.
        use_augmentation: If True, apply data augmentation to training.
        use_sampler: If True, apply a weighted sampler to the training loader.

    Returns:
        A dict mapping 'train', 'val', and 'test' to DataLoader objects.

    Raises:
        KeyError: If dataset_name is not in the registry.
        RuntimeError: For data loading or splitting failures.
    """
    # Verify dataset registration
    if dataset_name not in DATASET_REGISTRY:
        raise KeyError(f"Dataset '{dataset_name}' not found in registry.")

    # Data augmentation transformations (more transformations can be added)
    augmentation_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),        # resize images to be img_size x img_size
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),                          # transform data into Tensor
        transforms.Normalize(mean=mean, std=std)        # Normalize the data
    ])
    
    # Non-data augmentation transformations
    basic_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),        # resize images to be img_size x img_size
        transforms.ToTensor(),                          # transform data into Tensor
        transforms.Normalize(mean=mean, std=std)        # Normalize the data
    ])
    
    # Select the transformation that will be performed for thr training set (val and test are not augmented)
    train_transform = augmentation_transform if use_augmentation else basic_transform
    val_transform = basic_transform
    test_transform = basic_transform

    # Determine dataset structure
    loaders: Dict[str, DataLoader] = {}

    # Determine how the data will be loaded
    try:
        dataset_cls = DATASET_REGISTRY[dataset_name]
        root_dir = dataset_cls.getDir()

        # Check for pre-split folders
        has_pre_split = all(os.path.isdir(os.path.join(root_dir, split)) for split in ['train', 'val'])

        if has_pre_split:
            # Load split datasets
            train_ds = dataset_cls(mode='train', transform=train_transform)
            val_ds = dataset_cls(mode='val', transform=val_transform)
            sampler: Optional[Sampler] = get_weighted_sampler(train_ds) if use_sampler else None
            loaders['train'] = DataLoader(
                train_ds.dataset, batch_size=batch_size,
                sampler=sampler, shuffle=not use_sampler
            )
            loaders['val'] = DataLoader(val_ds.dataset, batch_size=batch_size, shuffle=True)

            # Optional test split
            test_dir = os.path.join(root_dir, 'test')
            if os.path.isdir(test_dir):
                test_ds = dataset_cls(mode='test', transform=test_transform)
                loaders['test'] = DataLoader(test_ds.dataset, batch_size=batch_size, shuffle=False)
        else:
            # Manual split
            logging.info("No pre-split folders found; performing manual split.")
            full_ds = dataset_cls(transform=basic_transform)
            total = len(full_ds)
            n_train = int(train_split * total)
            n_test = int(test_split * total)
            n_val = total - n_train - n_test

            train_subset, val_subset, test_subset = random_split(full_ds, [n_train, n_val, n_test])
            # Apply augmentation to train
            train_subset.dataset.transform = train_transform
            sampler = get_weighted_sampler(train_subset) if use_sampler else None
            loaders['train'] = DataLoader(
                train_subset.dataset, batch_size=batch_size,
                sampler=sampler, shuffle=not use_sampler
            )
            loaders['val'] = DataLoader(val_subset.dataset, batch_size=batch_size, shuffle=True)
            loaders['test'] = DataLoader(test_subset.dataset, batch_size=batch_size, shuffle=False)

        # Log stats
        _log_all_statistics(loaders)
        return loaders

    except Exception as e:
        logging.error(f"Failed to load data for '{dataset_name}': {e}")
        raise RuntimeError(f"Error in load_data: {e}")
