import os
import time
import logging
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_models(
    model_name: str,
    data_loaders: Dict[str, DataLoader],
    save_dir: str,
    learning_rate: float = 0.001,
    epochs: int = 10,
    optimizer: str = "adam",
    freeze_base: bool = True
):
    """
    Train a model using transfer learning with checkpointing and logging.

    Parameters:
        model_name (str): Name of the pre-trained model to use (e.g., 'resnet18', 'mobilenet_v2').
        data_loaders Dict[str, DataLoader]: DataLoaders containing data splits (e.g. train, val, test)
        save_dir (str): Directory to save the trained model and checkpoints.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        optimizer (str): Optimizer to use ('adam' or 'sgd').
        freeze_base (bool): Whether to freeze the base model layers.
    """
    # Load dataset
    logging.info("Loading datasets...")
    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]
    logging.info("Datasets loaded successfully.")
    
    # Select pre-trained model
    logging.info(f"Initializing pre-trained model: {model_name}")
    if model_name.lower() == "mobilenet_v2":
        model = models.mobilenet_v2(weights='DEFAULT')
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, len(train_loader.dataset.classes))
    elif model_name.lower() == "resnet18":
        model = models.resnet18(weights='DEFAULT')
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(train_loader.dataset.classes))
    else:
        logging.error(f"Unsupported model: {model_name}")
        raise ValueError(f"Unsupported model: {model_name}")

    # Freeze base layers if specified
    if freeze_base:
        logging.info("Freezing base layers of the model.")
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "fc"):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logging.info(f"Model moved to device: {device}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate) if optimizer.lower() == "adam" else SGD(
        model.parameters(), lr=learning_rate, momentum=0.9
    )

    # Initialize variables for checkpointing
    best_acc = 0.0
    best_model_path = os.path.join(save_dir, f"{model_name}_best.pth")
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Training will save checkpoints to: {save_dir}")

    # Training loop
    logging.info("Starting training...")
    since = time.time()

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        logging.info("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()               # Set model to training mode
                data_loader = train_loader
            else:
                model.eval()                # Set model to evaluate mode
                data_loader = val_loader

            running_loss = 0.0
            correct = 0
            total = 0

            with tqdm(total=len(data_loader), desc=f"{phase.capitalize()} Epoch {epoch + 1}/{epochs}") as pbar:
                # Iterate over data
                for inputs, labels in data_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()                                   # Zero the parameter gradients

                    # Forward (track history if only in train)
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                    # Update the progress bar
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                    pbar.update(1)

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = correct / total

            logging.info(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Checkpoint for the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"Checkpoint: Best model saved with accuracy {best_acc:.4f}")

    time_elapsed = time.time() - since
    logging.info(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    logging.info(f"Best validation accuracy: {best_acc:.4f}")

    # Load the best model weights before returning
    model.load_state_dict(torch.load(best_model_path))
    logging.info(f"Best model weights loaded from: {best_model_path}")
    return model


# if __name__ == "__main__":
