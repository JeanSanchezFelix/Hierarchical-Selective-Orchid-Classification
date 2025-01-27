# Datasets Module

The `datasets` module provides dataset classes for loading and preprocessing image datasets for training, validation, and testing. Each dataset class adheres to the PyTorch `Dataset` interface, enabling easy integration with PyTorch's `DataLoader`.

---

## Folder Structure
```
datasets/
├── __init__.py
├── CpAnemiaDataset.py
├── MonkeypoxDataset.py
├── registry.py
├── SkinCancerDataset.py
```

## Contents

- [Overview of Datasets Module](#overview-of-datasets-module)
- [Available Datasets](#available-datasets)
  - [CpAnemiaDataset](#cpanemiadataset)
  - [MonkeypoxDataset](#monkeypoxdataset)
  - [SkinCancerDataset](#skincancerdataset)
- [Dataset Registry](#dataset-registry)
- [How to Use](#how-to-use)
- [Examples](#examples)

---

## Overview of Datasets Module

The `datasets` module includes classes for handling:
- Loading image datasets from specified directories.
- Transforming datasets for preprocessing (e.g., resizing, normalization).
- Supporting train, validation, and test splits for structured datasets.
- Providing class labels and class-to-index mappings.

---

## Available Datasets

### `CpAnemiaDataset`

#### Description:
Handles the Cp-Anemia dataset, which is stored under `data/cp-anemia/download/`.

#### Features:
- Supports preprocessing transformations via `transform`.
- Provides class-to-index mappings (`class_to_idx`).

#### Methods:
- `__len__`: Returns the size of the dataset.
- `__getitem__`: Fetches a specific data sample by index.
- `getName`: Returns the name of the dataset.
- `getDir`: Returns the root directory of the dataset.

#### Tests:
The `TestCpAnemiaDataset` class validates:
- Total number of images.
- Number of classes.
- Class-wise image counts.

---

### `MonkeypoxDataset`

#### Description:
Handles the Monkeypox dataset, which is stored under `data/monkeypox/download/`.

#### Features:
- Supports train and test modes via the `mode` parameter.
- Applies preprocessing transformations through `transform`.
- Provides class-to-index mappings (`class_to_idx`).

#### Methods:
- `__len__`: Returns the size of the dataset.
- `__getitem__`: Fetches a specific data sample by index.
- `getName`: Returns the name of the dataset.
- `getDir`: Returns the root directory of the dataset.

#### Tests:
The `TestMonkeypoxDataset` class validates:
- Total number of images in train and test splits.
- Number of classes.
- Class-wise image counts for both train and test splits.

---

### `SkinCancerDataset`

#### Description:
Handles the Skin Lesions dataset, stored under `data/skin-lesions/download/`.

#### Features:
- Supports train, validation, and test modes via the `mode` parameter.
- Applies preprocessing transformations through `transform`.
- Provides class-to-index mappings (`class_to_idx`).

#### Methods:
- `__len__`: Returns the size of the dataset.
- `__getitem__`: Fetches a specific data sample by index.
- `getName`: Returns the name of the dataset.
- `getDir`: Returns the root directory of the dataset.

#### Tests:
The `TestSkinCancerDataset` class validates:
- Total number of images for train, validation, and test splits.
- Number of classes (14).
- Class-wise image counts for each split.

---

## Dataset Registry

The `registry.py` file provides a `DATASET_REGISTRY`, which maps dataset names to their respective classes for easy instantiation.

### `DATASET_REGISTRY`

| Dataset Name | Class             |
|--------------|-------------------|
| `CpAnemia`   | `CpAnemiaDataset` |
| `MonkeyPox`  | `MonkeypoxDataset`|
| `SkinCancer` | `SkinCancerDataset`|

---

## How to Use

### Example: Loading a Dataset

1. Import the dataset:
   ```python
   from datasets import CpAnemiaDataset
   from torchvision import transforms
   ```

2. Define preprocessing transformations:
   ```python
   transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
   ])
   ```

3. Load the dataset:
   ```python
   dataset = CpAnemiaDataset(transform=transform)
   ```

4. Use with a DataLoader:
   ```python
   from torch.utils.data import DataLoader

   data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
   ```

### Example: Using the Dataset Registry

1. Import the registry:
   ```python
   from datasets.registry import DATASET_REGISTRY
   ```

2. Instantiate a dataset using the registry:
   ```python
   dataset_class = DATASET_REGISTRY["CpAnemia"]
   dataset = dataset_class(transform=transform)
   ```

---

For more details, refer to the source code in the `datasets` folder.

