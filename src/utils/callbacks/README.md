# Callbacks Module

The `callbacks` module provides a set of utilities to manage various aspects of training workflows in PyTorch. Callbacks are designed to be hooks that are executed at specific points during training, such as at the start or end of an epoch, to enable features like early stopping, model checkpointing, and learning rate scheduling.

---

## Folder Structure
```
callbacks/
├── __init__.py
├── callbacks.py 
├── registry.py 
```

## Contents

- [Overview of Callbacks](#overview-of-callbacks)
- [Available Callbacks](#available-callbacks)
  - [Base Class: `Callback`](#base-class-callback)
  - [`ModelCheckpoint`](#modelcheckpoint)
  - [`EarlyStopping`](#earlystopping)
  - [`ReduceLROnPlateau`](#reducelronplateau)
- [Callback Registry](#callback-registry)
- [How to Use Callbacks](#how-to-use-callbacks)
- [Examples](#examples)

---

## Overview of Callbacks

Callbacks are modular and reusable components used to perform specific actions during training, such as:

- Saving the model when a monitored metric improves.
- Stopping training if a monitored metric stops improving for a specified number of epochs.
- Adjusting the learning rate dynamically when the monitored metric plateaus.

The module includes:
- `callbacks.py`: Defines the callback classes.
- `registry.py`: Provides a registry to manage callback instantiation.

---

## Available Callbacks

### Base Class: `Callback`

The `Callback` class is a base class for all callbacks. It defines the following hook methods, which are intended to be overridden by subclasses:

- `on_epoch_start(epoch: int, logs: dict = None)`: Triggered at the start of an epoch.
- `on_epoch_end(epoch: int, logs: dict = None)`: Triggered at the end of an epoch.
- `on_train_start(logs: dict = None)`: Triggered at the start of training.
- `on_train_end(logs: dict = None)`: Triggered at the end of training.
- `on_validation_start(logs: dict = None)`: Triggered at the start of validation.
- `on_validation_end(logs: dict = None)`: Triggered at the end of validation.

---

### `ModelCheckpoint`

#### Description:
Saves the model during training whenever a monitored metric improves.

#### Parameters:
- `monitor (str)`: The metric to monitor (e.g., `val_loss`).
- `save_best_only (bool)`: If `True`, only the best model is saved.
- `mode (str)`: One of `{'min', 'max'}` to determine whether a decrease or increase in the monitored metric is considered an improvement.
- `filepath (str)`: Path to save the model checkpoint.
- `verbose (bool)`: If `True`, logs when a model is saved.

---

### `EarlyStopping`

#### Description:
Stops training when a monitored metric stops improving for a specified number of epochs.

#### Parameters:
- `monitor (str)`: The metric to monitor (e.g., `val_loss`).
- `patience (int)`: The number of epochs to wait before stopping.
- `min_delta (float)`: Minimum change to qualify as an improvement.
- `mode (str)`: One of `{'min', 'max'}`.
- `save_path (str)`: File path to save the best model.
- `verbose (bool)`: If `True`, logs messages when Early Stop is triggered.

---

### `ReduceLROnPlateau`

#### Description:
Reduces the learning rate when a monitored metric stops improving.

#### Parameters:
- `monitor (str)`: The metric to monitor (e.g., `val_loss`).
- `factor (float)`: Factor by which the learning rate is reduced.
- `patience (int)`: Number of epochs to wait before reducing the learning rate.
- `min_delta (float)`: Minimum change to qualify as an improvement.
- `mode (str)`: One of `{'min', 'max'}`.
- `min_lr (float)`: Minimum allowable learning rate.
- `verbose (bool)`: If `True`, logs when the learning rate is reduced.

---

## Callback Registry

The `registry.py` file defines a `CALLBACK_REGISTRY`, which maps callback names to their constructors, and a utility function `process_callbacks()` for instantiating callbacks.

### `CALLBACK_REGISTRY`
A dictionary mapping callback names (`str`) to their corresponding classes:
- `"EarlyStopping"`
- `"ReduceLROnPlateau"`
- `"ModelCheckpoint"`

### `process_callbacks(callback_names: list, save_dir: str)`
#### Parameters:
- `callback_names (list)`: Names of the callbacks to instantiate.
- `save_dir (str)`: Directory to save checkpoint files.

#### Returns:
A list of instantiated callback objects.

---

## How to Use Callbacks

1. **Import Callbacks:**
   ```python
   from src.utils.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
   ```

2. **Use with Training Workflow:**
   Pass instantiated callbacks to your training script and invoke their hooks at appropriate points (e.g., `on_epoch_end`).

3. **Using the Registry:**
   You can use `process_callbacks()` to streamline callback instantiation.

---

## Examples

### Basic Usage
```python
from src.utils.callbacks import ModelCheckpoint

# Initialize a ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    filepath="checkpoints/best_model.pth",
    verbose=True
)

# Call it during training
for epoch in range(num_epochs):
    logs = {"val_loss": compute_val_loss(), "model": model}
    checkpoint_callback.on_epoch_end(epoch, logs)
```

### Using `process_callbacks`
```python
from src.utils.callbacks.registry import process_callbacks

# Define the callbacks to use
callback_names = ["EarlyStopping", "ModelCheckpoint"]
callbacks = process_callbacks(callback_names, save_dir="checkpoints")

# Use callbacks in your training loop
for epoch in range(num_epochs):
    logs = {"val_loss": compute_val_loss(), "model": model}
    for callback in callbacks:
        callback.on_epoch_end(epoch, logs)
        if hasattr(callback, "early_stop") and callback.early_stop:
            print("Early stopping triggered.")
            break
```
---

For more information, refer to the source code in `callbacks.py` and `registry.py`.

