# Utils Folder Documentation

The `utils` folder contains essential modules that support the functionality of the main script. Each file is responsible for a specific part of the machine learning pipeline, ensuring modularity and reusability. Below is an overview of the scripts and their key functions.

---

## Folder Structure
```
utils/
├── callbacks.py 
├── eval.py
├── metrics.py
├── parsing.py
├── preprocessing.py
├── setup.py
├── train.py
```

### Overview
1. **`callbacks.py`**: Contains regularization techniques for the training process (eg. EarlyStopping, etc.)
2. **`eval.py`**: TODO
3. **`metrics.py`**: Contains functions for plotting the performance results
4. **`parsing.py`**: Handles command-line argument parsing and configuration management.
5. **`preprocessing.py`**: Manages data loading, preprocessing, and augmentation.
6. **`train.py`**: Implements the training process, including checkpointing and performance evaluation.

---

## Script Details

### `callbacks.py`
#### Purpose
- Contains regularization techniques that prevent overfitting and bias in training
- Some of the techniques include earlystopping, etc.

#### Key Classes
1. **`EarlyStopping`**:
- 

#### Usage

---

### `eval.py`
#### Purpose
- 

#### Key Functions
1. 
- 

#### Usage

---

### `metrics.py`
#### Purpose
- 

#### Key Functions
1. 
- 

#### Usage

---

### `parsing.py`
#### Purpose
- This script is responsible for handling command-line arguments and optionally loading configurations from a CSV file.
- Provides default values and validates user inputs.

#### Key Functions
1. **`load_args_from_file(file_path)`**:
   - Reads configurations from a CSV file.
   - Returns a dictionary containing argument names and values.

2. **`parse()`**:
   - Parses command-line arguments and integrates them with optional configuration files.
   - Validates input values (e.g., dataset path, model parameters).
   - Returns a dictionary of configurations.

#### Usage
Used by `main.py` to parse and validate user-provided arguments or configuration files.

---

### `preprocessing.py`
#### Purpose
- Prepares the dataset for training, validation, and testing.
- Handles tasks like data splitting, augmentation, and normalization.

#### Key Functions
1. **`load_data(dataset_dir, batch_size, train_split, test_split, img_size, use_augmentation)`**:
   - Loads images from the specified directory.
   - Splits the dataset into training, validation, and testing subsets (if needed).
   - Applies transformations, including resizing, normalization, and optional augmentation.
   - Returns a dictionary of `DataLoader` objects for each data split (`train`, `val`, and optionally `test`).

2. **`log_dataset_statistics(dataset, dataset_name)`**:
   - Logs the size and class distribution of a dataset.
   - Useful for understanding dataset characteristics.

3. **`log_all_statistics(loaders)`**:
   - Logs statistics for all data splits (training, validation, and testing).

#### Usage
Used by `main.py` to load and preprocess data before training.

---

### `eval.py`
#### Purpose
- 

#### Key Functions
1. 
- 

#### Usage

---

### `train.py`
#### Purpose
- Facilitates the training of machine learning models using transfer learning.
- Includes checkpointing and performance monitoring.

#### Key Functions
1. **`train_models(model_name, data_loaders, save_dir, learning_rate, epochs, optimizer, freeze_base)`**:
   - Trains a model using a specified pre-trained architecture (e.g., ResNet, MobileNet).
   - Supports freezing base layers for transfer learning.
   - Saves the best-performing model based on validation accuracy.
   - Logs training and validation progress.

2. **Additional Features**:
   - Checkpointing: Saves the model with the best validation accuracy.
   - Performance metrics: Logs loss and accuracy for each epoch.
   - Device management: Utilizes GPU if available.

#### Usage
Called by `main.py` to perform model training based on user-provided configurations.

---

## Summary of Usage
The `utils` folder enables a seamless pipeline by breaking down the workflow into three clear steps:
1. **Parsing**:
   - Configurations and arguments are handled in `parsing.py`.
   - Ensures all necessary inputs are valid and accessible.

2. **Preprocessing**:
   - Data preparation tasks like splitting, augmentation, and normalization are handled in `preprocessing.py`.

3. **Training**:
   - Model training, checkpointing, and performance monitoring are implemented in `train.py`.

These scripts are designed to work together with the `main.py` file to provide a complete machine learning workflow.

---

For detailed examples of how these scripts interact, refer to the documentation for `main.py` or consult the function docstrings within each script.

