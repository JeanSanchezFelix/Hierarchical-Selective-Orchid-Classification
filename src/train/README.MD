# Train Module

The `train` module facilitates the training of deep learning models with PyTorch, leveraging transfer learning, custom callbacks, and evaluation utilities. This script is designed to handle training, validation, and optionally testing, while integrating seamlessly with other components of the repository.

---

## Folder Structure
```
train/
├── __init__.py
├── train.py
```

## Contents

- [Overview of Train Module](#overview-of-train-module)
- [Available Functions](#available-functions)
  - [train_models](#train_models)
  - [train_and_evaluate](#train_and_evaluate)
- [How to Use](#how-to-use)
- [Examples](#examples)

---

## Overview of Train Module

The `train` module provides utilities for:
- Training deep learning models using transfer learning.
- Utilizing callbacks for checkpoints, early stopping, and learning rate scheduling.
- Logging metrics and saving visualizations for performance analysis.
- Evaluating the model on validation and test datasets.

---

## Available Functions

### `train_models`

#### Description:
Trains a deep learning model with customizable parameters and callbacks.

#### Parameters:
- `model_name (str)`: Name of the pre-trained model (e.g., `resnet18`, `mobilenet_v2`).
- `data_loaders (dict[str, DataLoader])`: DataLoaders for training, validation, and optionally testing.
- `save_dir (str)`: Directory to save the trained model and checkpoints.
- `learning_rate (float)`: Learning rate for the optimizer (default: `0.001`).
- `epochs (int)`: Number of epochs for training (default: `5`).
- `criterion (str)`: Loss function (e.g., `cross_entropy`, `bce`).
- `optimizer (str)`: Optimizer to use (e.g., `adam`, `sgd`).
- `callbacks (list)`: List of callback instances (e.g., `ModelCheckpoint`, `EarlyStopping`).
- `pretrained_weights (str)`: Path to pre-trained weights (optional).
- `freeze_base (bool)`: If `True`, freezes the base model layers.

#### Returns:
- `model (nn.Module)`: The trained model.

---

### `train_and_evaluate`

#### Description:
Handles the core training and evaluation loop, tracking metrics and triggering callbacks.

#### Parameters:
- `model (nn.Module)`: The model to train.
- `train_loader (DataLoader)`: DataLoader for training data.
- `val_loader (DataLoader)`: DataLoader for validation data.
- `learning_rate (float)`: Learning rate for logging.
- `criterion (nn.Module)`: Loss function.
- `optimizer (torch.optim.Optimizer)`: Optimization algorithm.
- `epochs (int)`: Number of epochs.
- `device (torch.device)`: Device for training (`cpu` or `cuda`).
- `callbacks (list)`: List of callback instances.

#### Returns:
- `loss_dict (dict[str, list])`: Loss values for each phase across epochs.

---

## How to Use

### Example: Training a Model

1. Import the `train_models` function:
   ```python
   from src.train.train import train_models
   ```

2. Prepare DataLoaders and callbacks:
   ```python
   from src.utils.preprocessing import load_data
   from src.utils.callbacks import ModelCheckpoint, EarlyStopping

   data_loaders = load_data(dataset="SkinCancer", batch_size=32, use_augmentation=True)

   callbacks = [
       ModelCheckpoint(monitor="val_loss", save_best_only=True, mode="min", filepath="checkpoints/best_model.pth"),
       EarlyStopping(monitor="val_loss", patience=5, mode="min")
   ]
   ```

3. Train the model:
   ```python
   trained_model = train_models(
       model_name="mobilenet_v2",
       data_loaders=data_loaders,
       save_dir="checkpoints",
       learning_rate=0.001,
       epochs=10,
       criterion="cross_entropy",
       optimizer="adam",
       callbacks=callbacks,
       freeze_base=True
   )
   ```

---

## Examples

### Basic Training Loop
```python
from src.train.train import train_models
from src.utils.preprocessing import load_data
from src.utils.callbacks.registry import process_callbacks

# Load data
data_loaders = load_data(dataset="SkinCancer", batch_size=32)

# Initialize callbacks
callbacks = process_callbacks(callback_names=["EarlyStopping", "ModelCheckpoint"], save_dir="checkpoints")

# Train the model
trained_model = train_models(
    model_name="resnet18",
    data_loaders=data_loaders,
    save_dir="checkpoints",
    learning_rate=0.001,
    epochs=20,
    criterion="cross_entropy",
    optimizer="adam",
    callbacks=callbacks
)
```

---

For more details, refer to the source code in `train.py`.

