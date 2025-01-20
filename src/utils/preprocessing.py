import os
import logging
from collections import Counter
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def log_dataset_statistics(dataset, dataset_name: str):
    """
    Logs the size and class distribution of a dataset.

    Parameters:
        dataset: The dataset object (e.g., ImageFolder or a split subset).
        dataset_name (str): A descriptive name for the dataset (e.g., 'train', 'val', 'test').
    """
    logging.info(f"{dataset_name.capitalize()} dataset size: {len(dataset)}")
    if hasattr(dataset, 'dataset'):  # Handle subsets like random_split outputs
        classes = dataset.dataset.classes
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        classes = dataset.classes
        targets = dataset.targets

    class_counts = Counter(targets)
    logging.info(f"Class distribution for {dataset_name} dataset:")
    for class_idx, count in sorted(class_counts.items()):
        logging.info(f"  Class '{classes[class_idx]}': {count} samples")


def log_all_statistics(loaders: dict[str, DataLoader]):
    """
    Logs statistics for all DataLoaders.

    Parameters:
        loaders: A dictionary of DataLoaders (e.g., {'train': train_loader, 'val': val_loader}).
    """
    total_samples = 0
    for dataset_name, loader in loaders.items():
        dataset = loader.dataset
        log_dataset_statistics(dataset, dataset_name)
        total_samples += len(dataset)
    logging.info(f"Total dataset size: {total_samples}")

def load_data(
    dataset_dir: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    test_split: float = 0.1,
    img_size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    use_augmentation: bool = False
) -> dict[str, DataLoader]:
    """
    Load and preprocess data from a directory, handling both pre-split and unsplit datasets.

    Parameters:
        dataset_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for DataLoader.
        train_split (float): Proportion of data to use for training (if splitting).
        test_split (float): Proportion of data to use for testing (if splitting).
        img_size (int): Target size for image resizing.
        mean (tuple[float, float, float]): Mean values for normalization.
        std (tuple[float, float, float]): Standard deviation values for normalization.
        use_augmentation (bool): Whether to apply data augmentation to the training dataset.

    Returns:
        dict[str, DataLoader]: A dictionary containing DataLoaders for 'train', 'val', and optionally 'test'.
    """
    # Validate dataset directory
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory {dataset_dir} does not exist.")
    
    # Define transforms

    # Data augmentation transformations (more transformations can be added)
    augmentation_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),        # resize images to be img_size x img_size
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),                          # transform data into Tensor
        transforms.Normalize(mean=mean, std=std)        # Normalize the data
    ])
    
    # Non-data augmentation transformations
    basic_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),        # resize images to be img_size x img_size
        transforms.ToTensor(),                          # transform data into Tensor
        transforms.Normalize(mean=mean, std=std)        # Normalize the data
    ])
    
    # Select the transformation that will be performed for thr training set (val and test are not augmented)
    train_transforms = augmentation_transforms if use_augmentation else basic_transforms
    val_transforms = basic_transforms
    test_transforms = basic_transforms

    # Determine dataset structure
    loaders = {}

    # Determine how the data will be loaded
    if all(os.path.isdir(os.path.join(dataset_dir, subdir)) for subdir in ['train', 'val']):
        # Pre-split dataset: train and val directories exist
        train_dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, 'train'), transform=train_transforms)
        val_dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, 'val'), transform=val_transforms)
        loaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        loaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Check for a 'test' folder
        test_dir = os.path.join(dataset_dir, 'test')
        if os.path.isdir(test_dir):
            test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
            loaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    else:
        # Unsplit dataset: split manually
        logging.info("Detected unsplit dataset structure. Splitting dataset...")
        dataset = datasets.ImageFolder(root=dataset_dir, transform=basic_transforms)
        total_samples = len(dataset)
        train_size = int(train_split * total_samples)
        test_size = int(test_split * total_samples)
        val_size = total_samples - train_size - test_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Apply augmentation to train dataset and create dataset
        train_dataset.dataset.transform = train_transforms  
        loaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        loaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        loaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Log all dataset statistics
    log_all_statistics(loaders)

    return loaders

# if __name__ == "__main__":
