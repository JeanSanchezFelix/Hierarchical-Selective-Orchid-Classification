# Utils Module

The `utils` module contains essential utilities that support various aspects of the machine learning workflow, including evaluation, metric calculations, and model setup. These tools are designed to streamline the development process by providing reusable and modular components.

---

## Folder Structure
```
utils/
├── callbacks/
├── __init__.py
├── eval.py
├── metrics.py
├── model_setup.py
├── parsing.py
├── preprocessing.py
```

## Contents

- [Overview of Utils](#overview-of-utils)
- [Available Scripts](#available-scripts)
  - [eval.py](#evalpy)
  - [metrics.py](#metricspy)
  - [model_setup.py](#model_setuppy)
  - [parsing.py](#parsingpy)
  - [preprocessing.py](#preprocessingpy)
- [How to Use](#how-to-use)
- [Examples](#examples)

---

## Overview of Utils

The `utils` module provides foundational tools for tasks such as:
- Performing evaluation on test datasets.
- Computing and visualizing performance metrics.
- Configuring models, loss functions, and optimizers.
- Parsing arguments and loading configuration files.
- Preprocessing datasets for training and evaluation.

---

## Available Scripts

### eval.py

The `eval.py` script handles the evaluation of trained models on a test dataset. It supports tasks such as loading pre-trained models, running inference, and generating evaluation metrics and visualizations.

#### Functions:

- **`evaluate(model_path: str, img_size: int, dataset: str, save_dir: str = None)`**:
  - Performs inference and evaluates model performance.
  - Parameters:
    - `model_path`: Path to the saved model file.
    - `img_size`: Image size for resizing inputs.
    - `dataset`: Name of the dataset to evaluate.
    - `save_dir`: Directory to save evaluation plots (optional).

- **`compute_loss_and_predictions(outputs, labels, criterion=None)`**:
  - Computes loss (if criterion is provided) and generates predictions.
  - Parameters:
    - `outputs`: Raw model outputs (logits).
    - `labels`: Ground truth labels.
    - `criterion`: Loss function. If None, loss is not computed.

- **`test_inference(model: nn.Module, test_loader: DataLoader, device: torch.device, save_dir: str)`**:
  - Runs inference on the test dataset and computes metrics.
  - Parameters:
    - `model`: The trained PyTorch model.
    - `test_loader`: DataLoader for the test dataset.
    - `device`: Device (CPU/GPU) for inference.
    - `save_dir`: Directory to save generated plots (optional).

---

### metrics.py

The `metrics.py` script provides utilities for calculating performance metrics and visualizing them using various plots.

#### Functions:

- **`calculate_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]`**:
  - Calculates metrics like accuracy, recall, precision, F1-score, and AUC.
  - Supports both binary and multiclass classification.

- **`plot_metric_bar(metrics: dict, title: str = "Performance Metrics", save_path: str = None)`**:
  - Generates a bar chart for the given metrics.

- **`plot_confusion_matrix(y_true, y_pred, labels=None, title: str = "Confusion Matrix", save_path: str = None)`**:
  - Plots a confusion matrix as a heatmap.

- **`plot_train_val_curve(metrics: dict, metric_name: str = "Loss", title: str = "Training vs Validation", save_path: str = None)`**:
  - Plots training and validation metrics over epochs.

- **`plot_radar_chart(metrics, title="Radar Chart for Metrics", save_path=None)`**:
  - Creates a radar chart to visualize multiple metrics.

---

### model_setup.py

The `model_setup.py` script provides tools for configuring models, loss functions, and optimizers.

#### Functions:

- **`setup_model(model_name: str, pretrained_weights: bool, num_classes: int) -> nn.Module`**:
  - Sets up a pre-trained model with a customized classification head.
  - Supported models include MobileNet, ResNet, VGG, EfficientNet, and more.

- **`setup_criterion(criterion: str) -> nn.Module`**:
  - Configures the loss function for training.
  - Supported loss functions: CrossEntropy, MSE, L1, NLL, BCE, BCEWithLogits.

- **`setup_optimizer(model, optimizer: str, learning_rate: float)`**:
  - Configures the optimizer for training.
  - Supported optimizers: Adam, SGD, RMSprop, Adagrad, AdamW.

- **`training_setup(model_name: str, learning_rate: float, criterion: str, optimizer: str, pretrained_weights: bool, num_classes: int)`**:
  - Provides a complete setup for the model, optimizer, and loss function.
  - Returns the configured model, loss function, and optimizer.

---

### parsing.py

The `parsing.py` script provides utilities for parsing command-line arguments and configuration files. It supports flexible configuration management and logging.

#### Functions:

- **`load_args_from_file(file_path: str) -> dict[str, str]`**:
  - Reads arguments from a configuration file (CSV, JSON, YAML) and returns them as a dictionary.

- **`configure_logging(logs: bool, save_dir: str)`**:
  - Configures logging to write to a file and optionally to the console.

- **`validate_args(args: dict)`**:
  - Validates parsed arguments to ensure they meet expected conditions.

- **`parse() -> dict[str, int | str | list]`**:
  - Parses command-line arguments and configuration file inputs for training models.
  - Returns a dictionary of parsed and validated configuration values.

---

### preprocessing.py

The `preprocessing.py` script handles dataset preprocessing, including data augmentation, normalization, and splitting datasets into training, validation, and test sets.

#### Functions:

- **`log_dataset_statistics(dataset, dataset_name: str)`**:
  - Logs the size and class distribution of a dataset.

- **`log_all_statistics(loaders: dict[str, DataLoader])`**:
  - Logs statistics for all DataLoaders.

- **`load_data(dataset: str, batch_size: int = 32, train_split: float = 0.8, test_split: float = 0.1, img_size: int = 224, mean: tuple[float, float, float] = (0.485, 0.456, 0.406), std: tuple[float, float, float] = (0.229, 0.224, 0.225), use_augmentation: bool = False) -> dict[str, DataLoader]`**:
  - Loads and preprocesses data, handling both pre-split and unsplit datasets.
  - Returns a dictionary containing DataLoaders for 'train', 'val', and optionally 'test'.

---

## How to Use

### Example: Model Evaluation

1. Import the `evaluate` function:
   ```python
   from model_compression.src.utils.eval import evaluate
   ```

2. Call `evaluate` with the appropriate parameters:
   ```python
   evaluate(
       model_path="path/to/saved_model.pth",
       img_size=224,
       dataset="SkinCancer",
       save_dir="evaluation_outputs"
   )
   ```

### Example: Metric Visualization

1. Import the plotting functions:
   ```python
   from model_compression.src.utils.metrics import plot_metric_bar, plot_confusion_matrix
   ```

2. Use the functions with your metrics:
   ```python
   metrics = {"Accuracy": 0.95, "Precision": 0.92, "Recall": 0.93, "F1-Score": 0.94}
   plot_metric_bar(metrics, save_path="metrics_bar_chart.png")
   ```

### Example: Model Setup

1. Import the `training_setup` function:
   ```python
   from model_compression.src.utils.model_setup import training_setup
   ```

2. Configure the training components:
   ```python
   model, criterion, optimizer = training_setup(
       model_name="mobilenet_v2",
       learning_rate=0.001,
       criterion="cross_entropy",
       optimizer="adam",
       pretrained_weights=True,
       num_classes=3
   )
   ```

### Example: Argument Parsing

1. Import the `parse` function:
   ```python
   from model_compression.src.utils.parsing import parse
   ```

2. Parse arguments and load configuration:
   ```python
   config = parse()
   ```

### Example: Data Preprocessing

1. Import the `load_data` function:
   ```python
   from model_compression.src.utils.preprocessing import load_data
   ```

2. Load and preprocess data:
   ```python
   loaders = load_data(
       dataset="SkinCancer",
       batch_size=32,
       train_split=0.8,
       test_split=0.1,
       img_size=224,
       use_augmentation=True
   )
   ```

---

For more details, refer to the source code in the `utils` folder.

