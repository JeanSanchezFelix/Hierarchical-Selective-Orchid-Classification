# Main Script Documentation

This markdown file provides a detailed explanation of the `main.py` script, its purpose, and how it works. The script acts as the entry point for training machine learning models with customizable configurations provided via command-line arguments or configuration files.

---

## Overview

The `main.py` script orchestrates four main tasks:
1. **Parsing Command-line Arguments**: Uses the `utils.parsing.parse` function to handle configuration settings for the training process.
2. **Data Loading and Preprocessing**: Leverages the `utils.preprocessing` module to prepare datasets for training, validation, and testing.
3. **Model Training**: Uses the `train.train_models` function to train models with transfer learning and save checkpoints.
4. **Evaluation**: Optionally evaluates the trained model using utilities from `utils.eval` and generates performance metrics.

---

## Usage Instructions

### Command-line Arguments
The script supports the following arguments (handled by `utils.parsing.parse`):

- **Model Parameters:**
  - `--model_name`: Name of the pre-trained model (e.g., `resnet18`, `mobilenet_v2`).
  - `--pretrained_weights`: Path to pre-trained weights (default: `None`).
  - `--img_size`: Image size for input to the model (e.g., 224 for 224x224 images).

- **Training Parameters:**
  - `--epochs`: Number of training epochs (default: 5).
  - `--batch_size`: Batch size for training (default: 32).
  - `--learning_rate`: Learning rate for the optimizer (default: 0.001).
  - `--criterion`: Loss function (choices: `cross_entropy`, `mse`, `l1`, `nll`, `bce`, `bce_with_logits`; default: `cross_entropy`).
  - `--optimizer`: Optimizer to use (choices: `adam`, `sgd`, `rmsprop`, `adagrad`, `adamw`; default: `adam`).

- **Callbacks Parameters:**
  - `--callbacks`: List of callbacks to use (choices: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`; default: `ModelCheckpoint`).
  - `--ModelCheckpoint_monitor`: Metric to monitor for `ModelCheckpoint` (default: `val_loss`).
  - `--ModelCheckpoint_save_best_only`: Save only the best model (default: True).
  - `--ModelCheckpoint_mode`: Mode for `ModelCheckpoint` (choices: `min`, `max`; default: `min`).
  - `--ModelCheckpoint_verbose`: Enable verbose logging for `ModelCheckpoint` (default: False).
  - `--EarlyStopping_monitor`: Metric to monitor for `EarlyStopping` (default: `val_loss`).
  - `--EarlyStopping_patience`: Number of epochs to wait before stopping (default: 5).
  - `--EarlyStopping_min_delta`: Minimum change to qualify as improvement (default: 0.0).
  - `--EarlyStopping_mode`: Mode for `EarlyStopping` (choices: `min`, `max`; default: `min`).
  - `--EarlyStopping_verbose`: Enable verbose logging for `EarlyStopping` (default: False).
  - `--ReduceLROnPlateau_monitor`: Metric to monitor for `ReduceLROnPlateau` (default: `val_loss`).
  - `--ReduceLROnPlateau_factor`: Factor by which to reduce learning rate (default: 0.1).
  - `--ReduceLROnPlateau_patience`: Number of epochs to wait before reducing LR (default: 5).
  - `--ReduceLROnPlateau_min_delta`: Minimum change to qualify as improvement (default: 0.0).
  - `--ReduceLROnPlateau_mode`: Mode for `ReduceLROnPlateau` (choices: `min`, `max`; default: `min`).
  - `--ReduceLROnPlateau_min_lr`: Minimum learning rate for `ReduceLROnPlateau` (default: 1e-6).
  - `--ReduceLROnPlateau_verbose`: Enable verbose logging for `ReduceLROnPlateau` (default: False).

- **Data Parameters:**
  - `--dataset`: Name of the dataset to use (choices: `CpAnemia`, `MonkeyPox`, `SkinCancer`).
  - `--train_split`: Proportion of data to use for training (default: 0.8).
  - `--test_split`: Proportion of data to use for testing (default: 0.1).
  - `--data_augmentation`: Enable data augmentation (default: False).

- **Miscellaneous:**
  - `--save_dir`: Directory to save models and outputs (default: `./models`).
  - `--config_file`: Path to a YAML, CSV, or JSON configuration file.
  - `--logging`: Enable logging to the console (default: False).

### Example Usage

#### Using Command-line Arguments
```bash
python main.py \
    --model_name mobilenet_v2 \
    --epochs 10 \
    --dataset SkinCancer \
    --batch_size 32 \
    --learning_rate 0.001 \
    --train_split 0.8 \
    --test_split 0.1 \
    --img_size 224 \
    --data_augmentation 1 \
    --callbacks ModelCheckpoint EarlyStopping \
    --ModelCheckpoint_monitor val_loss \
    --ModelCheckpoint_save_best_only \
    --EarlyStopping_patience 5 \
    --ReduceLROnPlateau_factor 0.1 \
    --save_dir ./saved_models
```

#### Using a Configuration File
```bash
python main.py --config_file config.yaml
```

#### Combining Configuration File and Command-line Arguments
```bash
python main.py --config_file config.yaml \
    --epochs 20 \
    --learning_rate 0.0005
```
- In this example, values from `config.yaml` will be overridden by the `epochs` and `learning_rate` arguments provided via the command line.

---

## Folder Structure

```
repository/
├── main.py
├── data/
│   ├── cp-anemia/
│   ├── monkeypox/
│   ├── skin-lesions/
├── datasets/
│   ├── CpAnemiaDataset.py
│   ├── MonkeypoxDataset.py
│   ├── SkinCancerDataset.py
│   ├── registry.py
├── models/
├── notebooks/
├── src/
│   ├── train/
│   │   ├── train.py
│   ├── utils/
│       ├── callbacks/
│       │   ├── callbacks.py
│       │   ├── registry.py
│       ├── eval.py
│       ├── metrics.py
│       ├── model_setup.py
│       ├── parsing.py
│       ├── preprocessing.py
├── saved_models/
├── requirements.txt
```

### Key Components
- **`main.py`**: Entry point for running the training process.
- **`datasets/`**: Contains dataset classes and a registry for loading datasets.
- **`src/utils/`**: Provides utility scripts for callbacks, evaluation, metrics, model setup, argument parsing, and preprocessing.
- **`src/train/`**: Contains the `train.py` script for training models.
- **`saved_models/`**: Directory for storing trained models and outputs.

---

## Dependencies

Ensure the following Python libraries are installed:
- `torch`
- `torchvision`
- `tqdm`
- `numpy`
- `matplotlib`
- `seaborn`
- `PyYAML`
- `scikit-learn`

Install missing dependencies using:
```bash
pip install -r requirements.txt
```
or
```bash
pip install torch torchvision tqdm numpy matplotlib seaborn PyYAML scikit-learn
```

---

## Notes

- Ensure that datasets are correctly structured under the `data/` directory.
- Callback configurations can be customized in the command-line arguments or configuration file.
- For additional details, refer to the individual modules:
  - [Train Module Documentation](#train_readme)
  - [Utils Module Documentation](#utils_readme)
  - [Datasets Module Documentation](#datasets_readme).

