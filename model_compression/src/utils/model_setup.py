import os
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from model_compression.src.utils.data_imbalance import calculate_model_weights

def setup_model(
    model_name: str, 
    pretrained_weights_path: Optional[str], 
    num_classes: int,
    use_imagenet_weights: bool = True,
    binary_single_logit: bool = False,
) -> nn.Module:
    """
    Set up a pre-trained model with a custom classification head and optionally load custom weights.
    
    Depending on the provided `model_name`, this function selects and customizes a model architecture.
    When binary_single_logit is true for a two-class BCE task, the classification layer outputs
    a single logit; otherwise, it outputs logits for the specified number of classes.
    If a valid path is given in `pretrained_weights`, these weights are loaded into the model.
    
    Args:
        model_name: Identifier of the torchvision model (e.g., 'resnet18', 'mobilenet_v2').
        pretrained_weights_path: Optional path to custom weights file. If provided, load these weights.
        num_classes: Number of output classes.
        binary_single_logit: Use one output logit only for a two-class BCE task.
        
    Returns:
        A torchvision model with its classifier head adapted to `num_classes`.
    
    Raises:
        ValueError: If `model_name` is not supported.
        FileNotFoundError: If `pretrained_weights_path` is specified but the file does not exist.
    """
    # Determine whether to load custom weights or use default pre-trained weights.
    # If a valid custom weights path is provided, use random initialization (None) to allow loading later.
    # Otherwise, use the default pre-trained weights.
    use_custom_weights = bool(pretrained_weights_path and pretrained_weights_path.strip())
    weights_arg = None if use_custom_weights or not use_imagenet_weights else 'DEFAULT'
    
    # Configure the model based on the model name
    model_name_lower = model_name.lower()
    if model_name_lower == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights_arg)
        input_features = model.classifier[1].in_features
        output_features = 1 if binary_single_logit and num_classes == 2 else num_classes
        model.classifier[1] = nn.Linear(input_features, output_features)
    elif model_name_lower == "resnet18":
        model = models.resnet18(weights=weights_arg)
        input_features = model.fc.in_features
        output_features = 1 if binary_single_logit and num_classes == 2 else num_classes
        model.fc = nn.Linear(input_features, output_features)
    elif model_name_lower in ["resnet34", "resnet50", "resnet101", "resnet152"]:
        model = getattr(models, model_name_lower)(weights=weights_arg)
        input_features = model.fc.in_features
        output_features = 1 if binary_single_logit and num_classes == 2 else num_classes
        model.fc = nn.Linear(input_features, output_features)
    elif model_name_lower in ["mobilenet_v3_large", "mobilenet_v3_small"]:
        model = getattr(models, model_name_lower)(weights=weights_arg)
        input_features = model.classifier[0].in_features
        output_features = 1 if binary_single_logit and num_classes == 2 else num_classes
        model.classifier[0] = nn.Linear(input_features, output_features)
    elif model_name_lower in ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
                              "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"]:
        model = getattr(models, model_name_lower)(weights=weights_arg)
        input_features = model.classifier[1].in_features
        output_features = 1 if binary_single_logit and num_classes == 2 else num_classes
        model.classifier[1] = nn.Linear(input_features, output_features)
    elif model_name_lower in ["vgg11", "vgg13", "vgg16", "vgg19", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]:
        model = getattr(models, model_name_lower)(weights=weights_arg)
        input_features = model.classifier[0].in_features
        output_features = 1 if binary_single_logit and num_classes == 2 else num_classes
        model.classifier[0] = nn.Linear(input_features, output_features)
    elif model_name_lower in ["densenet121", "densenet161", "densenet169", "densenet201"]:
        model = getattr(models, model_name_lower)(weights=weights_arg)
        input_features = model.classifier.in_features
        output_features = 1 if binary_single_logit and num_classes == 2 else num_classes
        model.classifier = nn.Linear(input_features, output_features)
    elif model_name_lower in ["squeezenet1_0", "squeezenet1_1"]:
        model = getattr(models, model_name_lower)(weights=weights_arg)
        # For SqueezeNet, the classifier is a Sequential containing a Conv2d layer.
        in_channels = model.classifier[1].in_channels
        # When num_classes == 2, output a single channel; otherwise, use num_classes channels.
        model.classifier[1] = nn.Conv2d(in_channels, 1 if binary_single_logit and num_classes == 2 else num_classes, kernel_size=(1, 1))
    elif model_name_lower in ["shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0"]:
        model = getattr(models, model_name_lower)(weights=weights_arg)
        input_features = model.fc.in_features
        output_features = 1 if binary_single_logit and num_classes == 2 else num_classes
        model.fc = nn.Linear(input_features, output_features)
    elif model_name_lower == "inception_v3":
        model = models.inception_v3(weights=weights_arg)
        input_features = model.fc.in_features
        output_features = 1 if binary_single_logit and num_classes == 2 else num_classes
        model.fc = nn.Linear(input_features, output_features)
    elif model_name_lower == "alexnet":
        model = models.alexnet(weights=weights_arg)
        input_features = model.classifier[1].in_features
        output_features = 1 if binary_single_logit and num_classes == 2 else num_classes
        model.classifier[1] = nn.Linear(input_features, output_features)
    else:
        logging.error(f"Unsupported model: {model_name}")
        raise ValueError(f"Unsupported model: {model_name}")
    
    # If a custom weights file is provided, load the weights into the model.
    if use_custom_weights:
        if not os.path.isfile(pretrained_weights_path):
            logging.error(f"Weights file not found: {pretrained_weights_path}")
            raise FileNotFoundError(f"Weights file not found: {pretrained_weights_path}")
        state_dict = torch.load(pretrained_weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        logging.info(f"Loaded custom weights from {pretrained_weights_path}")
    
    return model

def setup_criterion(
    criterion_name: str,
    dataloader: DataLoader,
    use_class_weights: bool = False
) -> nn.Module:
    """
    Create a loss function, optionally with class weighting for imbalanced datasets.

    Args:
        criterion_name: Loss identifier ('cross_entropy', 'mse', 'bce', etc.).
        dataloader: DataLoader to compute class weights from labels if requested.
        use_class_weights: Whether to apply class weights.

    Returns:
        A configured PyTorch loss module.

    Raises:
        ValueError: If `criterion_name` is unsupported.
    """
    # Pre-compute class weights if requested.
    weights = calculate_model_weights(dataloader) if use_class_weights else None
    name = criterion_name.lower()

    if name == "cross_entropy":
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif name == "mse":
        criterion = nn.MSELoss(weight=weights)
    elif name == "l1":
        criterion = nn.L1Loss()  # Note: L1Loss does not support a weight parameter.
    elif name == "nll":
        criterion = nn.NLLLoss(weight=weights)
    elif name == "bce":
        criterion = nn.BCELoss(weight=weights)
    elif name == "bce_with_logits":
        criterion = nn.BCEWithLogitsLoss(weight=weights)
    else:
        logging.error(f"Unsupported criterion_name: {criterion_name}")
        raise ValueError(f"Unsupported criterion_name: {criterion_name}")

    return criterion

def setup_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float
) -> torch.optim.Optimizer:
    """
    Initialize an optimizer for the given model parameters.

    Args:
        model: PyTorch model whose parameters will be optimized.
        optimizer_name: Optimizer identifier ('adam', 'sgd', etc.).
        learning_rate: Learning rate for the optimizer.

    Returns:
        An instantiated optimizer.

    Raises:
        ValueError: If `optimizer_name` is unsupported.
    """
    name = optimizer_name.lower()

    if name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif name == "rmsprop":
        opt = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.9)
    elif name == "adagrad":
        opt = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        logging.error(f"Unsupported optimizer_name: {optimizer_name}")
        raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")

    return opt

def tf_setup(
    model_name: str,
    pretrained_weights_path: Optional[str],
    data_loader: DataLoader,
    criterion_name: str,
    optimizer_name: str,
    learning_rate: float,
    use_class_weights: bool = False
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """
    Full setup for model, loss, and optimizer.

    Args:
        model_name: Name of the pretrained model.
        pretrained_weights_path: Optional path to custom weights.
        dataloader: DataLoader for training (used to infer number of classes).
        criterion_name: Loss function identifier.
        optimizer_name: Optimizer identifier.
        learning_rate: Learning rate.
        use_class_weights: Whether to apply class weights in loss.

    Returns:
        A tuple of (model, criterion, optimizer).
    """
    # Configure model, criterion, and optimizer
    num_classes = len(data_loader.dataset.classes)
    model = setup_model(
        model_name, pretrained_weights_path, num_classes,
        binary_single_logit=criterion_name.lower() in {"bce", "bce_with_logits"},
    )
    criterion= setup_criterion(criterion_name, data_loader, use_class_weights)
    optimizer = setup_optimizer(model, optimizer_name, learning_rate)
    return model, criterion, optimizer
