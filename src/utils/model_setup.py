import logging
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchvision import models
from src.utils.data_imbalance import calculate_model_weights

def setup_model(model_name: str, pretrained_weights: bool, num_classes: int) -> nn.Module:
    """
    Set up a pre-trained model with a customized classification head.

    Parameters:
        model_name (str): Name of the pre-trained model (e.g., "resnet18", "mobilenet_v2").
        pretrained_weights (bool): If True, load custom weights; otherwise, use default pre-trained weights.
        num_classes (int): Number of output classes for the classification task.

    Returns:
        nn.Module: The configured model with a modified classification head.

    Raises:
        ValueError: If the specified model_name is unsupported.
    """
    # Select the model based on the name and configure its classification layer
    if model_name.lower() == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None if pretrained_weights else 'DEFAULT')
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 1) if num_classes == 2 else nn.Linear(num_features, num_classes)
    elif model_name.lower() == "resnet18":
        model = models.resnet18(weights=None if pretrained_weights else 'DEFAULT')
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name.lower() in ["resnet34", "resnet50", "resnet101", "resnet152"]:
        model = getattr(models, model_name.lower())(weights=None if pretrained_weights else 'DEFAULT')
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name.lower() in ["mobilenet_v3_large", "mobilenet_v3_small"]:
        model = getattr(models, model_name.lower())(weights=None if pretrained_weights else 'DEFAULT')
        num_features = model.classifier[0].in_features
        model.classifier[0] = nn.Linear(num_features, num_classes)
    elif model_name.lower() in ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
                                "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"]:
        model = getattr(models, model_name.lower())(weights=None if pretrained_weights else 'DEFAULT')
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    elif model_name.lower() in ["vgg11", "vgg13", "vgg16", "vgg19", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]:
        model = getattr(models, model_name.lower())(weights=None if pretrained_weights else 'DEFAULT')
        num_features = model.classifier[0].in_features
        model.classifier[0] = nn.Linear(num_features, num_classes)
    elif model_name.lower() in ["densenet121", "densenet161", "densenet169", "densenet201"]:
        model = getattr(models, model_name.lower())(weights=None if pretrained_weights else 'DEFAULT')
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    elif model_name.lower() in ["squeezenet1_0", "squeezenet1_1"]:
        model = getattr(models, model_name.lower())(weights=None if pretrained_weights else 'DEFAULT')
        model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, num_classes, kernel_size=(1, 1))
    elif model_name.lower() in ["shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0"]:
        model = getattr(models, model_name.lower())(weights=None if pretrained_weights else 'DEFAULT')
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name.lower() == "inception_v3":
        model = models.inception_v3(weights=None if pretrained_weights else 'DEFAULT')
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name.lower() == "alexnet":
        model = models.alexnet(weights=None if pretrained_weights else 'DEFAULT')
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    else:
        logging.error(f"Unsupported model: {model_name}")
        raise ValueError(f"Unsupported model: {model_name}")

    return model

def setup_criterion(criterion: str, dataloader: torch.utils.data.DataLoader, class_weights: bool) -> nn.Module:
    """
    Configure the loss function for training.

    Parameters:
        criterion (str): Name of the loss function (e.g., "cross_entropy", "mse").

    Returns:
        nn.Module: The configured loss function.

    Raises:
        ValueError: If the specified criterion is unsupported.
    """
    # Map the criterion name to the corresponding loss function
    if criterion.lower() == "cross_entropy":
        criterion = nn.CrossEntropyLoss(weight=calculate_model_weights(dataloader) if class_weights else None)
    elif criterion.lower() == "mse":
        criterion = nn.MSELoss(weight=calculate_model_weights(dataloader) if class_weights else None)
    elif criterion.lower() == "l1":
        criterion = nn.L1Loss(weight=calculate_model_weights(dataloader) if class_weights else None)
    elif criterion.lower() == "nll":
        criterion = nn.NLLLoss(weight=calculate_model_weights(dataloader) if class_weights else None)
    elif criterion.lower() == "bce":
        criterion = nn.BCELoss(weight=calculate_model_weights(dataloader) if class_weights else None)
    elif criterion.lower() == "bce_with_logits":
        criterion = nn.BCEWithLogitsLoss(weight=calculate_model_weights(dataloader) if class_weights else None)
    else:
        logging.error(f"Unsupported loss function: {criterion}")
        raise ValueError(f"Unsupported loss function: {criterion}")
    
    return criterion

def setup_optimizer(model, optimizer: str, learning_rate: float):
    """
    Configure the optimizer for training.

    Parameters:
        model (nn.Module): The model whose parameters need optimization.
        optimizer (str): Name of the optimizer (e.g., "adam", "sgd").
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        torch.optim.Optimizer: The configured optimizer.

    Raises:
        ValueError: If the specified optimizer is unsupported.
    """
    # Map the optimizer name to the corresponding optimizer
    if optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        logging.error(f"Unsupported optimizer: {optimizer}")
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    return optimizer

def training_setup(model_name: str, 
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
    criterion = setup_criterion(criterion, dataloader, class_weights)
    optimizer = setup_optimizer(model, optimizer, learning_rate)
    return model, criterion, optimizer
