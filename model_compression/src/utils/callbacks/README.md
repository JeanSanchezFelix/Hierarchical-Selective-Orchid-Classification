# Callbacks Module

The `callbacks` module provides utilities to manage training workflows in PyTorch through hookable components. Each callback executes specific logic at predefined points during training (e.g., epoch start/end), enabling features such as early stopping, checkpointing, learning-rate scheduling, and more.

---

## Folder Structure
```plaintext
callbacks/
├── __init__.py
├── callbacks.py       # Definitions of callback classes
└── registry.py        # Callback registry and factory 
```

---

## Contents

- [Overview](#modules-overview)
- [Available Callbacks](#available-callbacks)
  - [Base Class: `Callback`](#base-class-callback)
  - [`ModelCheckpoint`](#modelcheckpoint)
  - [`EarlyStopping`](#earlystopping)
  - [`ReduceLROnPlateau`](#reducelronplateau)
  - [`LRScheduler`](#lrscheduler)
- [Callback Registry](#callback-registry)
- [How to Use](#how-to-use)

---

## Modules Overview

Callbacks are modular, reusable components that hook into the training loop to perform tasks such as:

- Saving model checkpoints when metrics improve
- Stopping training early based on validation performance
- Dynamically adjusting the learning rate
- Logging custom metrics or behaviors

All callback classes inherit from the `Callback` base class defined in `callbacks.py`, which provides empty hook methods:

```python
class Callback:
    def on_train_start(self, logs=None): ...
    def on_epoch_start(self, epoch, logs=None): ...
    def on_validation_start(self, logs=None): ...
    def on_validation_end(self, logs=None): ...
    def on_epoch_end(self, epoch, logs=None): ...
    def on_train_end(self, logs=None): ...
```

---

## Available Callbacks

### Base Class: `Callback`
Abstract base for all callbacks; implements hook methods that receive a `logs` dictionary containing training state (e.g., metrics, model, optimizer).

---

### `ModelCheckpoint`

**Description:**
Saves the model’s state dict (and metadata) when a monitored metric improves.

**Constructor Arguments:**
- `monitor: str` — metric key to watch (default `'val_loss'`).
- `save_best_only: bool` — if `True`, overwrite only on improvement.
- `mode: {'min','max'}` — whether lower or higher values are better.
- `save_path: str` — file path for saving the checkpoint.
- `verbose: bool` — logs save events if `True`.

---

### `EarlyStopping`

**Description:**
Halts training if the monitored metric fails to improve for a set number of epochs.

**Constructor Arguments:**
- `monitor: str` — metric to watch.
- `patience: int` — epochs with no improvement before stopping.
- `min_delta: float` — minimal change to qualify as improvement.
- `mode: {'min','max'}`
- `save_path: str` — where to save the best model weights.
- `verbose: bool` — logs progress if `True`.

---

### `ReduceLROnPlateau`

**Description:**
Reduces the optimizer’s learning rate when the monitored metric plateaus.

**Constructor Arguments:**
- `monitor: str`
- `factor: float` — LR multiplier on plateau (e.g., `0.1`).
- `patience: int`
- `min_delta: float`
- `mode: {'min','max'}`
- `min_lr: float` — lower bound for LR.
- `verbose: bool`

---

### `LRScheduler`

**Description:**
Wraps any PyTorch LR scheduler (`_LRScheduler`) to call `scheduler.step()` each epoch.

**Constructor Arguments:**
- `scheduler: torch.optim.lr_scheduler._LRScheduler` — scheduler instance.
- `verbose: bool` — logs LR changes if `True`.

---

## Callback Registry

`registry.py` defines:

- **`CALLBACK_REGISTRY`** — maps names (`str`) to callback classes.
- **`process_callbacks(args)`** — factory that reads:
  - `args['callbacks']`: list of callback names, and
  - per-callback params in `args` (e.g., `'EarlyStopping_patience'`)

It returns a dict of instantiated callbacks ready to be passed into your training loop.

---

## How to Use

1. **Instantiate callbacks** via registry or directly:
   ```python
   from model_compression.src.utils.callbacks import EarlyStopping, ModelCheckpoint

   callbacks = [
       EarlyStopping(monitor='val_loss', patience=3, save_path='best.pth'),
       ModelCheckpoint(monitor='val_accuracy', mode='max', save_path='chkpt.pth')
   ]
   ```
2. **Attach to training loop:**
   ```python
   for cb in callbacks:
       cb.on_train_start()
   for epoch in range(epochs):
       for cb in callbacks:
           cb.on_epoch_start(epoch)
       # train...
       for cb in callbacks:
           cb.on_epoch_end(epoch, logs)
       if any(getattr(cb, 'early_stop', False) for cb in callbacks):
           break
   for cb in callbacks:
       cb.on_train_end()
   ```
3. **Or use `process_callbacks`:**
   ```python
   from model_compression.src.utils.callbacks.registry import process_callbacks

   config = {'callbacks': ['EarlyStopping','LRScheduler'],
             'EarlyStopping_patience': 5,
             'scheduler': my_scheduler}
   callbacks_map = process_callbacks(config)
   callbacks = list(callbacks_map.values())
   ```

---

See `callbacks.py` and `registry.py` for full API details and additional parameters.

