# Model Compression Engine

This repository contains the implementation of a model compression engine for biomedical applications, such as anemia detection through conjunctiva pallor, monkeypox classification, and skin lesion detection. The project includes pre-trained models, custom models, and utilities for training, evaluation, and preprocessing.

---

## Project Structure

```
repository/
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
├── main.py
├── requirements.txt
```

### Key Components

- **`data/`**: Contains the image datasets for various biomedical use cases:
  - **cp-anemia/**: Images for anemia detection through conjunctiva pallor.
  - **monkeypox/**: Images for monkeypox classification.
  - **skin-lesions/**: Images for skin cancer and other skin condition detection.

- **`datasets/`**: Contains dataset classes and a registry for easy instantiation. Includes:
  - `CpAnemiaDataset.py`: Handles the Cp-Anemia dataset.
  - `MonkeypoxDataset.py`: Handles the Monkeypox dataset.
  - `SkinCancerDataset.py`: Handles the Skin Lesions dataset.
  - `registry.py`: Maps dataset names to their respective classes.

- **`models/`**: Directory for model definitions (currently empty, for user customization).

- **`notebooks/`**: Jupyter notebooks for experimentation and analysis.

- **`src/`**:
  - **`train/`**: Contains the `train.py` script for model training.
  - **`utils/`**: Provides utility scripts for training and evaluation:
    - `callbacks/`: Contains callbacks for model checkpointing, early stopping, and learning rate scheduling.
    - `eval.py`: Handles model evaluation and metric calculation.
    - `metrics.py`: Provides custom metrics and visualization utilities.
    - `model_setup.py`: Configures models, optimizers, and loss functions.
    - `parsing.py`: Parses command-line arguments and configuration files.
    - `preprocessing.py`: Handles data loading, augmentation, and splitting.

- **`saved_models/`**: Directory to store trained models and checkpoints.

- **`main.py`**: Entry point for training and evaluation workflows.

- **`requirements.txt`**: Lists all required Python libraries and dependencies.

---

## How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/model-compression.git
cd model-compression
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train a Model
Run the `main.py` script to train and evaluate a model. Use command-line arguments or configuration files for customization.

#### Example Usage with Command-line Arguments
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

#### Example Usage with Configuration File
```bash
python main.py --config_file config.yaml
```

#### Example Usge Combining Configuration File and Command-line Arguments
```bash
python main.py --config_file config.yaml \
    --epochs 20 \
    --learning_rate 0.0005
```
- In this example, values from `config.yaml` will be overridden by the `epochs` and `learning_rate` arguments provided via the command line.

---

### 4. Explore Jupyter Notebooks
Use the notebooks in the `notebooks/` directory for further experimentation and analysis.

---

## Notes

- Ensure that datasets are correctly structured under the `data/` directory.
- The `main.py` script integrates various utilities, including callbacks, argument parsing, and model training, for an end-to-end training pipeline.
- Refer to the individual module documentation for detailed information on datasets, utilities, and training workflows.

