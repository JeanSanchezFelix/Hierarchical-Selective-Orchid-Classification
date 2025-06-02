# Train Module

The `train` module facilitates the training of deep learning models with PyTorch, supporting transfer learning, quantization-aware training (QAT), and knowledge distillation (KD). It integrates custom callbacks and evaluation utilities to streamline model development.

---

## Folder Structure
```
train/
├── __init__.py
├── train.py
├── knowledge_distillation.py
```

---

## Contents

- [Overview of Train Module](#modules-overview)
- [Available Functions](#available-functions)
  - [train_models](#transfer_learning)
  - [train_qat_with_kd](#train_qat_with_kd)
- [How to Use](#how-to-use)

---

## Modules Overview

This module supports two training workflows:

### 1. `train.py`
- Transfer learning using pretrained models.
- Custom callbacks (e.g., EarlyStopping, ModelCheckpoint).
- Metric logging and visualization.
- Evaluation on validation and test sets.

### 2. `knowledge_distillation.py`
- Quantization-aware training (QAT) combined with knowledge distillation (KD).
- Lightweight student model trained using a high-capacity teacher model.
- Callbacks and plotting included.

---

## Available Functions

### `transfer_learning`

#### Description:
Trains a deep learning model using transfer learning with custom callbacks.

#### Parameters:
- `model_name (str)`
- `data_loaders (dict[str, DataLoader])`
- `save_dir (str)`
- `learning_rate (float)`
- `epochs (int)`
- `criterion (str)`
- `optimizer (str)`
- `callbacks (list)`
- `pretrained_weights (str)`
- `freeze_base (bool)`

#### Returns:
- `model (nn.Module)`: Trained model.

---

### `train_qat_with_kd`

#### Description:
Trains a quantization-aware student model using knowledge distillation from a teacher model.

#### Parameters:
- `teacher_model (nn.Module)`
- `data_loaders (dict[str, DataLoader])`
- `save_dir (str)`
- `learning_rate (float)`
- `epochs (int)`
- `temperature (float)`
- `alpha (float)`
- `callbacks (list)`
- `quantized_model_path (str)`
- `export_onnx (bool)`
- `device (torch.device)`

#### Returns:
- `student_model (nn.Module)`: The trained student model.

---

## How to Use

### A. Transfer Learning (from `train.py`)
```python
from model_compression.src.train.train import transfer_learning
from model_compression.src.utils.preprocessing import load_data
from model_compression.src.utils.callbacks import ModelCheckpoint, EarlyStopping

data_loaders = load_data(dataset="SkinCancer", batch_size=32, use_augmentation=True)

callbacks = [
    ModelCheckpoint(monitor="val_loss", save_best_only=True, mode="min", filepath="checkpoints/best_model.pth"),
    EarlyStopping(monitor="val_loss", patience=5, mode="min")
]

trained_model = transfer_learning(
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

### B. Knowledge Distillation with QAT (from `knowledge_distillation.py`)
```python
from model_compression.src.train.knowledge_distillation import train_qat_with_kd
from model_compression.src.utils.preprocessing import load_data
from model_compression.src.utils import load_data, process_callbacks

dataset = "SkinCancer"
batch_size = 32
learning_rate = 0.001
epochs = 1
save_dir = f"models/{dataset}/Quantized"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model = "mobilenet_v2"
student_model = "mobilenet_v2"
args = {"callbacks": ["ModelCheckpoint", "EarlyStopping", "ReduceLROnPlateau"], 
        "save_dir": save_dir, 
        "model_name": student_model,
        "save_name": "qat_kd"}
CALLBACKS = list(process_callbacks(args).values())

os.makedirs(save_dir, exist_ok=True)

dataloaders = load_data(dataset=dataset, batch_size=batch_size)

teacher_model_weights = 'models/SkinCancer/mobilenet_v2_best_model.pth'

criterion = "cross_entropy"
optimizer = "adam"

teacher, quantized_student = train_qat_kd(
        teacher_name=teacher_model, 
        student_name=student_model,
        data_loaders=dataloaders,
        save_dir=save_dir,
        learning_rate=learning_rate,
        epochs=epochs,
        criterion = criterion,
        optimizer = optimizer,
        callbacks=CALLBACKS,
        quant_mode="export",
        teacher_model_weights=teacher_model_weights,
        device=device,
)
```

---

For more details, refer to the source code in `train.py` and `knowledge_distillation.py`.

