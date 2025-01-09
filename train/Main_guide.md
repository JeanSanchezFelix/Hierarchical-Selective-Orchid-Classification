# Main Script Documentation

This markdown file provides a detailed explanation of the `main.py` script, its purpose, and how it works. The script acts as the entry point for training machine learning models with customizable configurations provided via command-line arguments or configuration files.

## Overview
The `main.py` script orchestrates three main tasks:
1. **Parsing Command-line Arguments**: Handles configuration settings for the training process.
2. **Data Loading and Preprocessing**: Prepares the dataset for training and validation.
3. **Model Training**: Uses transfer learning to train models and save checkpoints.

---

## Usage Instructions

### Command-line Arguments
The script supports the following arguments (handled by `utils.parsing.parse`):
- **`--config_file`**: Path to a CSV configuration file.
- **`--model_name`**: Pre-trained model name (e.g., `resnet18`, `mobilenet_v2`).
- **`--num_models`**: Number of models to train.
- **`--num_epochs`**: Number of training epochs.
- **`--dataset_dir`**: Directory containing the dataset.
- **`--batch_size`**: Batch size for training.
- **`--learning_rate`**: Learning rate for the optimizer.
- **`--optimizer`**: Optimizer to use (`adam` or `sgd`).
- **`--train_split`**: Training split proportion (0.0 - 1.0).
- **`--test_split`**: Testing split proportion (0.0 - 1.0).
- **`--img_size`**: Image size for input to the model.
- **`--data_augmentation`**: Enable data augmentation (0 for False, 1 for True).
- **`--save_dir`**: Directory to save trained models.

### Example Usage
#### Using Command-line Arguments
```bash
python main.py \
    --model_name mobilenet_v2 \
    --num_epochs 10 \
    --dataset_dir /path/to/dataset \
    --batch_size 32 \
    --learning_rate 0.001 \
    --train_split 0.8 \
    --test_split 0.1 \
    --img_size 224 \
    --data_augmentation 1 \
    --save_dir ./saved_models
```

#### Using a Configuration File
```bash
python main.py --config_file /path/to/config.csv
```
- The CSV file should include the same keys as the command-line arguments.

---

## Folder Structure
```
repository/
├── main.py
├── utils/
│   ├── callbacks.py
│   ├── eval.py
│   ├── metrics.py
│   ├── parsing.py
│   ├── preprocessing.py
│   ├── train.py
└── saved_models/
```
- **`main.py`**: Entry point for running the training process.
- **`utils/`**: Contains utility scripts for parsing, preprocessing, and training.
- **`saved_models/`**: Directory where trained models are saved.

---

## Dependencies
Ensure the following Python libraries are installed:
- `torch`
- `torchvision`
- `tqdm`
- `numpy`
- `logging`

Install missing dependencies using:
```bash
pip install -r requirements.txt
```
or
```bash
pip install torch torchvision tqdm numpy
```

---

## Notes
- Always ensure the dataset directory exists and contains the required structure.
- For detailed configurations, refer to the `utils.parsing.parse` function.
