import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from typing import Optional
import torchvision.models.quantization as quant_models
from datasets.registry import DATASET_REGISTRY

import os
import logging
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

def load_data(
    dataset: str,
    batch_size: int = 32,
    img_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    use_augmentation: bool = False,
    use_sampler: bool = False,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Load and preprocess data, returning DataLoader objects for training, validation, and test splits.
    
    This function retrieves a dataset from a registry, applies the appropriate transformations 
    (with optional data augmentation), and constructs DataLoaders. If a weighted sampler is desired,
    it is applied to the training set. It also checks for a test split based on the dataset's directory.
    
    Args:
        dataset (str): The key to select the dataset from DATASET_REGISTRY.
        batch_size (int): Number of samples per batch.
        img_size (int): Target image size (height and width) for resizing.
        mean (Tuple[float, float, float]): Mean values for normalization.
        std (Tuple[float, float, float]): Standard deviation values for normalization.
        use_augmentation (bool): Whether to apply data augmentation on training data.
        use_sampler (bool): Whether to use a weighted sampler for the training data.
        num_workers (int): Number of subprocesses to use for data loading.
    
    Returns:
        Dict[str, DataLoader]: A dictionary containing DataLoader instances for 'train', 'val', and, if available, 'test'.
    
    Raises:
        KeyError: If the dataset key is not found in DATASET_REGISTRY.
    """
    # Validate that the dataset exists in the registry.
    if dataset not in DATASET_REGISTRY:
        logging.error(f"Dataset '{dataset}' not found in DATASET_REGISTRY.")
        raise KeyError(f"Dataset '{dataset}' not found in DATASET_REGISTRY.")
    
    # Define common transformation: resizing, tensor conversion, and normalization.
    resize_transform = transforms.Resize((img_size, img_size))
    basic_transforms = transforms.Compose([
        resize_transform,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Define augmentation transformations (applied only on training data if enabled).
    augmentation_transforms = transforms.Compose([
        resize_transform,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Select the appropriate transformation for the training set.
    train_transforms = augmentation_transforms if use_augmentation else basic_transforms
    val_transforms = basic_transforms
    test_transforms = basic_transforms
    
    # Retrieve the dataset directory and create dataset instances for each mode.
    root_dir = DATASET_REGISTRY[dataset].getDir()
    train_dataset = DATASET_REGISTRY[dataset](mode='train', transform=train_transforms)
    val_dataset = DATASET_REGISTRY[dataset](mode='val', transform=val_transforms)
    
    # If use_sampler is True, initialize a weighted sampler for the training dataset.
    sampler = get_weighted_sampler(train_dataset) if use_sampler else None

    # Create DataLoader instances.
    # If the dataset instance has a 'dataset' attribute (e.g., when wrapped), use it.
    train_loader = DataLoader(
        train_dataset.dataset if hasattr(train_dataset, "dataset") else train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset.dataset if hasattr(val_dataset, "dataset") else val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    loaders: Dict[str, DataLoader] = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Check if a test folder exists and, if so, create the test DataLoader.
    test_dir = os.path.join(root_dir, 'test')
    if os.path.isdir(test_dir):
        test_dataset = DATASET_REGISTRY[dataset](mode='test', transform=test_transforms)
        test_loader = DataLoader(
            test_dataset.dataset if hasattr(test_dataset, "dataset") else test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        loaders['test'] = test_loader

    # Log dataset statistics (ensure that log_all_statistics is defined appropriately).
    log_all_statistics(loaders)
    
    return loaders


def load_kd_model(model_name: str, pretrained_weights: Optional[str], num_classes: int) -> nn.Module:
    """
    Loads a model for knowledge distillation with support for quantization.
    
    If a valid pretrained_weights path is provided, it assumes the model is a teacher and
    loads the standard (non-quantized) version from torchvision.models, then applies the provided weights.
    Otherwise, it assumes the model is a student and loads the quantization-compatible version from
    torchvision.models.quantization with default weights.
    
    Supports the following models with quantized versions:
        - googlenet
        - inception_v3
        - mobilenet_v2
        - mobilenet_v3_large
        - resnet18
        - resnet50
        - resnext101_32x8d
        - shufflenet_v2_x0_5
        - shufflenet_v2_x1_0
        - shufflenet_v2_x1_5
        - shufflenet_v2_x2_0
        
    For binary classification (num_classes == 2), the final classification layer is modified to output a single logit;
    otherwise, it is set to match the number of classes.
    
    Args:
        model_name (str): Name of the model.
        pretrained_weights (Optional[str]): Path to saved weights. If provided, the teacher (standard model) is loaded.
        num_classes (int): Number of output classes.
        
    Returns:
        nn.Module: The configured model.
        
    Raises:
        FileNotFoundError: If a custom weights path is provided but does not exist.
        ValueError: If an unsupported model name is given.
    """
    # Determine whether a custom weights path is provided.
    use_custom_weights: bool = bool(pretrained_weights and pretrained_weights.strip())
    weights_arg = None if use_custom_weights else 'DEFAULT'
    model_name_lower = model_name.lower()
    
    # Supported models for quantization.
    supported_models = {
        "googlenet", "inception_v3", "mobilenet_v2", "mobilenet_v3_large",
        "resnet18", "resnet50", "resnext101_32x8d",
        "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0"
    }
    
    if model_name_lower not in supported_models:
        # logging.error(f"Unsupported model for quantization: {model_name}")
        raise ValueError(f"Unsupported model for quantization: {model_name}")
    
    # Load teacher (standard) model if custom weights provided; otherwise, load student (quantized) model.
    if use_custom_weights:
        # Teacher: use the standard models module.
        if model_name_lower == "googlenet":
            model = models.googlenet(weights=weights_arg)
        elif model_name_lower == "inception_v3":
            model = models.inception_v3(weights=weights_arg)
        elif model_name_lower == "mobilenet_v2":
            model = models.mobilenet_v2(weights=weights_arg)
        elif model_name_lower == "mobilenet_v3_large":
            model = models.mobilenet_v3_large(weights=weights_arg)
        elif model_name_lower == "resnet18":
            model = models.resnet18(weights=weights_arg)
        elif model_name_lower == "resnet50":
            model = models.resnet50(weights=weights_arg)
        elif model_name_lower == "resnext101_32x8d":
            model = models.resnext101_32x8d(weights=weights_arg)
        elif model_name_lower.startswith("shufflenet_v2"):
            model = getattr(models, model_name_lower)(weights=weights_arg)
    else:
        # Student: use the quantization models module.
        if model_name_lower == "googlenet":
            model = quant_models.googlenet(weights=weights_arg)
        elif model_name_lower == "inception_v3":
            model = quant_models.inception_v3(weights=weights_arg)
        elif model_name_lower == "mobilenet_v2":
            model = quant_models.mobilenet_v2(weights=weights_arg)
        elif model_name_lower == "mobilenet_v3_large":
            model = quant_models.mobilenet_v3_large(weights=weights_arg)
        elif model_name_lower == "resnet18":
            model = quant_models.resnet18(weights=weights_arg)
        elif model_name_lower == "resnet50":
            model = quant_models.resnet50(weights=weights_arg)
        elif model_name_lower == "resnext101_32x8d":
            model = quant_models.resnext101_32x8d(weights=weights_arg)
        elif model_name_lower.startswith("shufflenet_v2"):
            model = getattr(quant_models, model_name_lower)(weights=weights_arg)
    
    # Update the classification head based on the model architecture.
    # For binary classification (num_classes == 2), output a single logit; otherwise, output logits for all classes.
    if model_name_lower in ["googlenet", "inception_v3", "resnet18", "resnet50", "resnext101_32x8d",
                            "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0"]:
        # These models use a direct fc layer.
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1) if num_classes == 2 else nn.Linear(in_features, num_classes)
    elif model_name_lower in ["mobilenet_v2"]:
        # MobileNetV2 uses a sequential container for the classifier.
        in_features = model.last_channel
        model.classifier[1] = nn.Linear(in_features, 1) if num_classes == 2 else nn.Linear(in_features, num_classes)
    elif model_name_lower in ["mobilenet_v3_large"]:
        # MobileNetV3_large uses a sequential classifier, typically with the first layer as the linear.
        in_features = model.classifier[0].in_features
        model.classifier[0] = nn.Linear(in_features, 1) if num_classes == 2 else nn.Linear(in_features, num_classes)
    
    # If custom weights are provided, load them.
    if use_custom_weights:
        if os.path.exists(pretrained_weights):
            # Note: using map_location='cpu' for flexibility; adjust as needed.
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            model.load_state_dict(state_dict)
            # logging.info(f"Teacher weights loaded from: {pretrained_weights}")
        else:
            # logging.error(f"Custom weights path does not exist: {pretrained_weights}")
            raise FileNotFoundError(f"Custom weights file not found: {pretrained_weights}")
    
    return model