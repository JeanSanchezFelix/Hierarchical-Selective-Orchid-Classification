import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from typing import Optional

def prepare_dataloaders(data_directory: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """
    Prepares the training and evaluation dataloaders.
    
    Parameters:
        data_directory (str): Path to the dataset root directory.
        batch_size (int): Batch size for the dataloaders

    Returns:
        tuple: (train_loader, eval_loader)
    """
    # Define normalization parameters (using ImageNet values for MobileNetV2)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img_size = 224
    
    # Data augmentation and preprocessing for training
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # Preprocessing for evaluation (center crop)
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # Assuming the data directory has subfolders "train" and "val"
    train_dataset = datasets.ImageFolder(os.path.join(data_directory, 'train'),
                                         transform=train_transform)
    eval_dataset = datasets.ImageFolder(os.path.join(data_directory, 'val'),
                                        transform=eval_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, eval_loader

def load_mobilenetv2_model(weights_path: str, num_classes: int, device: torch.device, q_type: str = "post-training") -> nn.Module:
    """
    Loads the fine-tuned MobileNetV2 model for skin lesion classification.
    
    Parameters:
        weights_path (str): Path to the saved model weights.
        num_classes (int): The number of classes in the dataset.
    
    Returns:
        nn.Module: The loaded MobileNetV2 model.
    """
    # Load a MobileNetV2 pre-trained on ImageNet first
    if q_type == "post-training":
        model = models.mobilenet_v2(weights='DEFAULT')
    else:
        model = models.quantization.mobilenet_v2(weights='DEFAULT')
    
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    # Load the saved fine-tuned weights
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model

def load_model(model_name: str, pretrained_weights: Optional[str], num_classes: int) -> nn.Module:
    """
    Loads a model for knowledge distillation.
    
    Parameters:
        pretrained_weights (Optional[str]): Path to the saved model weights. If None, loads a quantized version.
        num_classes (int): The number of classes in the dataset.
        device (torch.device): Device to load the model on.
    
    Returns:
        nn.Module: The loaded model.
    """
    use_custom_weights = bool(pretrained_weights and pretrained_weights.strip())
    weights_arg = None if use_custom_weights else 'DEFAULT'

    # If pretrained_weights is provided, load the standard MobileNetV2, otherwise load the quantization-ready version.
    if pretrained_weights is not None:
        model = models.mobilenet_v2(weights=weights_arg)
    else:
        model = models.quantization.mobilenet_v2(weights=weights_arg)
    
    # Replace the classifier layer to adapt to the given number of classes.
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    # Load saved weights if a valid path is provided.
    if pretrained_weights is not None:
        state_dict = torch.load(pretrained_weights, weights_only=True)
        model.load_state_dict(state_dict)
    
    return model