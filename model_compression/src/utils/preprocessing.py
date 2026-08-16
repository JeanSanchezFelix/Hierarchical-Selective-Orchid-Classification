import os
import csv
import logging
from collections import Counter
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Sampler, Subset
from torchvision import transforms

from datasets.registry import DATASET_REGISTRY
from model_compression.src.utils.data_imbalance import get_weighted_sampler

class TransformSubset(Dataset):
    """
    Wraps a Subset and applies a transform without mutating the parent dataset.
    
    Args:
        subset (`torch.utils.data.Subset`): An existing Subset to wrap and apply transforms to.
        transform (`transforms.Compose`): A transformation object to apply to each image.  
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        root_ds = subset.dataset
        self.classes = root_ds.classes
        self.class_to_idx = root_ds.class_to_idx
        self.targets = [root_ds.targets[i] for i in subset.indices]

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        sample = self.subset[idx]

        if len(sample) == 3:
            img, label, weight = sample
        else:
            img, label = sample
            weight = None

        if self.transform:
            img = self.transform(img)
        return (img, label, weight) if weight is not None else (img, label)


def _manifest_subsets(
    dataset: Dataset,
    root_dir: str,
    manifest_path: str,
) -> Dict[str, Subset]:
    """Create train/validation/test subsets from a reviewed split manifest.

    The manifest must contain relative ``image_path`` and ``split`` columns.
    Image paths are matched against the dataset's underlying ImageFolder samples,
    so labels always remain owned by the configured dataset class.
    """
    image_folder = getattr(dataset, "dataset", dataset)
    samples = getattr(image_folder, "samples", None)
    if samples is None:
        raise ValueError("Manifest splits require a dataset exposing ImageFolder-style samples.")

    root = os.path.abspath(root_dir)
    path_to_index = {
        os.path.normcase(os.path.abspath(path)): index
        for index, (path, _) in enumerate(samples)
    }
    indices: Dict[str, list[int]] = {"train": [], "val": [], "test": []}
    assigned: set[int] = set()

    with open(manifest_path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"image_path", "split"}
        missing_columns = required - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"Split manifest is missing columns: {sorted(missing_columns)}")
        for row_number, row in enumerate(reader, start=2):
            split = (row.get("split") or "").strip()
            if split not in indices:
                raise ValueError(f"Unknown split '{split}' at manifest row {row_number}.")
            image_path = os.path.normcase(os.path.abspath(os.path.join(root, row["image_path"])))
            index = path_to_index.get(image_path)
            # A genus expert intentionally sees only a subset of a complete
            # manifest. Extra rows therefore are ignored, while every image
            # that *is* in the configured task must still be assigned below.
            if index is None:
                continue
            if index in assigned:
                raise ValueError(f"Manifest assigns the same image more than once: {row['image_path']}")
            assigned.add(index)
            indices[split].append(index)

    unassigned = len(samples) - len(assigned)
    if unassigned:
        raise ValueError(
            f"Manifest assigns {len(assigned)} of {len(samples)} dataset images; {unassigned} images are unassigned."
        )
    empty_splits = [split for split, values in indices.items() if not values]
    if empty_splits:
        raise ValueError(f"Manifest has empty required split(s): {empty_splits}")

    return {split: Subset(dataset, values) for split, values in indices.items()}
    
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
    use_sampler: bool = False,
    random_seed: int = 18,
    split_manifest: Optional[str] = None,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
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
        split_manifest: Optional CSV generated by ``orchid_split_audit.py``.
            When set, this overrides pre-split directories and random splitting.
        dataset_kwargs: Optional constructor arguments for the selected dataset.
            For TaxonomicOrchid this enables explicit ``task`` and
            ``target_genus`` modes without changing generic datasets.

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
        dataset_kwargs = dict(dataset_kwargs or {})
        root_dir = dataset_kwargs.get("root_dir") or dataset_cls.getDir()

        # A reviewed manifest must take precedence over implicit split logic.
        if split_manifest:
            logging.info("Using reviewed split manifest: %s", split_manifest)
            full_ds = dataset_cls(transform=None, **dataset_kwargs)
            manifest_subsets = _manifest_subsets(full_ds, root_dir, split_manifest)
            train_subset = TransformSubset(manifest_subsets["train"], train_transform)
            val_subset = TransformSubset(manifest_subsets["val"], basic_transform)
            test_subset = TransformSubset(manifest_subsets["test"], basic_transform)
            sampler = get_weighted_sampler(train_subset) if use_sampler else None
            loaders["train"] = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, shuffle=not use_sampler)
            loaders["val"] = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
            loaders["test"] = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        # Check for pre-split folders
        elif all(os.path.isdir(os.path.join(root_dir, split)) for split in ['train', 'val']):
            # Load split datasets
            train_ds = dataset_cls(mode='train', transform=train_transform, **dataset_kwargs)
            val_ds = dataset_cls(mode='val', transform=val_transform, **dataset_kwargs)
            sampler: Optional[Sampler] = get_weighted_sampler(train_ds) if use_sampler else None
            loaders['train'] = DataLoader(
                train_ds, batch_size=batch_size,
                sampler=sampler, shuffle=not use_sampler
            )
            loaders['val'] = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

            # Optional test split
            test_dir = os.path.join(root_dir, 'test')
            if os.path.isdir(test_dir):
                test_ds = dataset_cls(mode='test', transform=test_transform, **dataset_kwargs)
                loaders['test'] = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        else:
            # Manual split
            logging.info("No pre-split folders found; performing manual split.")
            full_ds = dataset_cls(transform=None, **dataset_kwargs)  # no transform at dataset level

            # Scrape targets and infer classes from full_ds if not exposed 
            if not hasattr(full_ds, 'classes') or full_ds.classes is None:
                if hasattr(full_ds, 'class_to_idx'):
                    idx_to_class = {v: k for k, v in full_ds.class_to_idx.items()}
                    full_ds.classes = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
                    logging.info(f"Inferred classes from class_to_idx: {full_ds.classes}")

            if not hasattr(full_ds, 'targets'):
                logging.info("Dataset has no 'targets' attribute, scraping labels...")
                full_ds.targets = [full_ds[i][1] for i in range(len(full_ds))]

            total = len(full_ds)
            n_train = int(train_split * total)
            n_test = int(test_split * total)
            n_val = total - n_train - n_test
            train_subset, val_subset, test_subset = random_split(
                full_ds, [n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(random_seed)
            )
            # Apply augmentation to training subset only, and basic transforms to val/test
            train_subset = TransformSubset(train_subset, train_transform)  
            val_subset   = TransformSubset(val_subset,   basic_transform)  
            test_subset  = TransformSubset(test_subset,  basic_transform) 

            sampler = get_weighted_sampler(train_subset) if use_sampler else None
            loaders['train'] = DataLoader(
                train_subset, batch_size=batch_size,
                sampler=sampler, shuffle=not use_sampler
            )
            loaders['val'] = DataLoader(val_subset,  batch_size=batch_size, shuffle=True)
            loaders['test'] = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        # Log stats
        _log_all_statistics(loaders)
        return loaders

    except Exception as e:
        logging.error(f"Failed to load data for '{dataset_name}': {e}")
        raise RuntimeError(f"Error in load_data: {e}")
