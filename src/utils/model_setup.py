import os
import logging
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchvision import models
from src.utils.data_imbalance import calculate_model_weights
from typing import Optional

def setup_model(model_name: str, pretrained_weights: Optional[str], num_classes: int) -> nn.Module:
    """
    Set up a pre-trained model with a custom classification head and optionally load custom weights.
    
    Depending on the provided `model_name`, this function selects and customizes a model architecture.
    For binary classification (i.e. num_classes == 2), the classification layer is modified to output
    a single logit; otherwise, it outputs logits for the specified number of classes.
    If a valid path is given in `pretrained_weights`, these weights are loaded into the model.
    
    Args:
        model_name (str): Name of the pre-trained model (e.g., "resnet18", "mobilenet_v2").
        pretrained_weights (Optional[str]): File path to the custom pre-trained weights.
            If a valid path is provided, these weights will be loaded. If None or an empty string is given,
            the model is initialized with default pre-trained weights.
        num_classes (int): Number of output classes for the classification task.
        
    Returns:
        nn.Module: Configured model with the customized classification head and optionally loaded weights.
    
    Raises:
        FileNotFoundError: If a custom weights path is provided but does not exist.
        ValueError: If the specified model_name is unsupported.
    """
    # Determine whether to load custom weights or use default pre-trained weights.
    # If a valid custom weights path is provided, use random initialization (None) to allow loading later.
    # Otherwise, use the default pre-trained weights.
    use_custom_weights = bool(pretrained_weights and pretrained_weights.strip())
    weights_arg = None if use_custom_weights else 'DEFAULT'
    
    # Configure the model based on the model name
    model_name_lower = model_name.lower()
    if model_name_lower == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights_arg)
        num_features = model.classifier[1].in_features
        # For binary classification, output a single logit; otherwise, match the number of classes.
        model.classifier[1] = nn.Linear(num_features, 1) if num_classes == 2 else nn.Linear(num_features, num_classes)
    elif model_name_lower == "resnet18":
        model = models.resnet18(weights=weights_arg)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1) if num_classes == 2 else nn.Linear(num_features, num_classes)
    elif model_name_lower in ["resnet34", "resnet50", "resnet101", "resnet152"]:
        model = getattr(models, model_name_lower)(weights=weights_arg)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1) if num_classes == 2 else nn.Linear(num_features, num_classes)
    elif model_name_lower in ["mobilenet_v3_large", "mobilenet_v3_small"]:
        model = getattr(models, model_name_lower)(weights=weights_arg)
        num_features = model.classifier[0].in_features
        model.classifier[0] = nn.Linear(num_features, 1) if num_classes == 2 else nn.Linear(num_features, num_classes)
    elif model_name_lower in ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
                              "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"]:
        model = getattr(models, model_name_lower)(weights=weights_arg)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 1) if num_classes == 2 else nn.Linear(num_features, num_classes)
    elif model_name_lower in ["vgg11", "vgg13", "vgg16", "vgg19", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]:
        model = getattr(models, model_name_lower)(weights=weights_arg)
        num_features = model.classifier[0].in_features
        model.classifier[0] = nn.Linear(num_features, 1) if num_classes == 2 else nn.Linear(num_features, num_classes)
    elif model_name_lower in ["densenet121", "densenet161", "densenet169", "densenet201"]:
        model = getattr(models, model_name_lower)(weights=weights_arg)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 1) if num_classes == 2 else nn.Linear(num_features, num_classes)
    elif model_name_lower in ["squeezenet1_0", "squeezenet1_1"]:
        model = getattr(models, model_name_lower)(weights=weights_arg)
        # For SqueezeNet, the classifier is a Sequential containing a Conv2d layer.
        in_channels = model.classifier[1].in_channels
        # When num_classes == 2, output a single channel; otherwise, use num_classes channels.
        model.classifier[1] = nn.Conv2d(in_channels, 1 if num_classes == 2 else num_classes, kernel_size=(1, 1))
    elif model_name_lower in ["shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0"]:
        model = getattr(models, model_name_lower)(weights=weights_arg)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1) if num_classes == 2 else nn.Linear(num_features, num_classes)
    elif model_name_lower == "inception_v3":
        model = models.inception_v3(weights=weights_arg)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1) if num_classes == 2 else nn.Linear(num_features, num_classes)
    elif model_name_lower == "alexnet":
        model = models.alexnet(weights=weights_arg)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 1) if num_classes == 2 else nn.Linear(num_features, num_classes)
    else:
        logging.error(f"Unsupported model: {model_name}")
        raise ValueError(f"Unsupported model: {model_name}")
    
    # If a custom weights file is provided, load the weights into the model.
    if use_custom_weights:
        if os.path.exists(pretrained_weights):
            # Load the custom weights from the file path
            state_dict = torch.load(pretrained_weights, map_location='cpu')
            model.load_state_dict(state_dict)
            logging.info(f"Custom pre-trained weights loaded from: {pretrained_weights}")
        else:
            logging.error(f"Custom weights path does not exist: {pretrained_weights}")
            raise FileNotFoundError(f"Custom weights file not found: {pretrained_weights}")
    
    return model

def setup_criterion(criterion_name: str, dataloader: DataLoader, use_class_weights: bool) -> nn.Module:
    """
    Configure and return the loss function for training.

    Parameters:
        criterion_name (str): Name of the loss function (e.g., "cross_entropy", "mse", "l1", "nll", "bce", "bce_with_logits").
        dataloader (DataLoader): DataLoader used for training (used to compute class weights if enabled).
        use_class_weights (bool): If True, computes and passes class weights to the loss function.

    Returns:
        nn.Module: The configured loss function.

    Raises:
        ValueError: If the specified criterion_name is unsupported.
    """
    # Pre-compute class weights if requested.
    weights = calculate_model_weights(dataloader) if use_class_weights else None
    criterion_name_lower = criterion_name.lower()

    if criterion_name_lower == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    elif criterion_name_lower == "mse":
        loss_fn = nn.MSELoss(weight=weights)
    elif criterion_name_lower == "l1":
        loss_fn = nn.L1Loss()  # Note: L1Loss does not support a weight parameter.
    elif criterion_name_lower == "nll":
        loss_fn = nn.NLLLoss(weight=weights)
    elif criterion_name_lower == "bce":
        loss_fn = nn.BCELoss(weight=weights)
    elif criterion_name_lower == "bce_with_logits":
        loss_fn = nn.BCEWithLogitsLoss(weight=weights)
    else:
        logging.error(f"Unsupported loss function: {criterion_name}")
        raise ValueError(f"Unsupported loss function: {criterion_name}")

    return loss_fn

def setup_optimizer(model, optimizer_name: str, learning_rate: float):
    """
    Configure and return the optimizer for training.

    Parameters:
        model (nn.Module): The model whose parameters need optimization.
        optimizer_name (str): Name of the optimizer (e.g., "adam", "sgd", "rmsprop", "adagrad", "adamw").
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        torch.optim.Optimizer: The configured optimizer.

    Raises:
        ValueError: If the specified optimizer_name is unsupported.
    """
    optimizer_name_lower = optimizer_name.lower()

    if optimizer_name_lower == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name_lower == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name_lower == "rmsprop":
        opt = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name_lower == "adagrad":
        opt = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_name_lower == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        logging.error(f"Unsupported optimizer: {optimizer_name}")
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return opt

def tf_setup(model_name: str, 
                learning_rate: float, 
                criterion: str, 
                optimizer: str, 
                pretrained_weights: bool, 
                dataloader: torch.utils.data.DataLoader,
                class_weights: bool
) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """
    Complete setup for the model, optimizer, and loss function.

    Parameters:
        model_name (str): Name of the pre-trained model to use.
        learning_rate (float): Learning rate for the optimizer.
        criterion (str): Name of the loss function.
        optimizer (str): Name of the optimizer.
        pretrained_weights (bool): If True, load custom weights; otherwise, use default pre-trained weights.
        dataloader (DataLoader): Loader for training data.

    Returns:
        tuple: Configured model, loss function, and optimizer.
    """
    # Configure model, criterion, and optimizer
    num_classes = len(dataloader.dataset.classes)
    model = setup_model(model_name, pretrained_weights, num_classes)
    criterion_fn = setup_criterion(criterion, dataloader, class_weights)
    optimizer_obj = setup_optimizer(model, optimizer, learning_rate)
    return model, criterion_fn, optimizer_obj