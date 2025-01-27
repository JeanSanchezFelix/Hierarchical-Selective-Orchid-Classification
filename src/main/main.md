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

- **`--config_file`**: Path to a CSV, JSON, or YAML configuration file.
- **`--model_name`**: Name of the pre-trained model (e.g., `resnet18`, `mobilenet_v2`).
- **`--epochs`**: Number of training epochs.
- **`--dataset`**: Dataset name (`CpAnemia`, `MonkeyPox`, or `SkinCancer`).
- **`--batch_size`**: Batch size for training.
- **`--learning_rate`**: Learning rate for the optimizer.
- **`--optimizer`**: Optimizer to use (e.g., `adam`, `sgd`).
- **`--train_split`**: Proportion of data to use for training (0.0 - 1.0).
- **`--test_split`**: Proportion of data to use for testing (0.0 - 1.0).
- **`--img_size`**: Image size for input to the model (e.g., 224 for 224x224 images).
- **`--data_augmentation`**: Enable data augmentation (0 for False, 1 for True).
- **`--callbacks`**: List of callbacks to use (e.g., `ModelCheckpoint`, `EarlyStopping`).
- **`--save_dir`**: Directory to save trained models and outputs.

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
    --save_dir ./saved_models
```

#### Using a Configuration File
```bash
python main.py --config_file /path/to/config.yaml
```
- The configuration file can be in CSV, JSON, or YAML format and should include keys matching the command-line arguments.

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

- Ensure the dataset directory structure matches the expected format for the datasets.
- Callback configurations can be customized in the command-line arguments or configuration file.
- For additional details, refer to the individual modules:
  - [Train Module Documentation](#train_readme)
  - [Utils Module Documentation](#utils_readme)
  - [Datasets Module Documentation](#datasets_readme).

