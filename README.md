# Model Compression for Biomedical Applications

This repository contains the implementation of a project focusing on model compression techniques applied to biomedical datasets, such as anemia detection through conjunctiva pallor, monkeypox classification, and skin lesion detection. The project includes both pre-trained and custom models, as well as relevant Jupyter notebooks for experimentation.

## Project Structure

- **data/**: This directory contains the image data for various biomedical use cases.
  - **cp-anemia/**: Images related to anemia detection through conjunctiva pallor.
    - `.gitignore`: Ignores unnecessary files.
    - `test.jpeg, test.jpg, test.png`: Example test images for anemia classification.
  - **monkeypox/**: Images used for classifying monkeypox.
    - `.gitignore`: Ignores unnecessary files.
    - `test.jpeg, test.jpg, test.png`: Example test images for monkeypox classification.
  - **skin-lesions/**: Skin lesion images used for detecting skin cancer or other conditions.
    - `.gitignore`: Ignores unnecessary files.
    - `test.jpeg, test.jpg, test.png`: Example test images for skin lesion classification.

- **main/**: Contains the main code for model training and evaluation.
  - `main.py`: The main script that runs the model training and compression tasks.

- **models/**: This folder contains the definitions of the models used or created for the project.
  - `models.py`: Defines the architecture of the models used in the project.

- **notebooks/**: Jupyter notebooks for experimentation and analysis.
  - `jupyter_notebook.ipynb`: Jupyter notebook containing code for data preprocessing, model training, and evaluation.
  - `jupyter_notebook-2.ipynb`: Additional notebook for further analysis and experimentation.

- **transfer_learning/**: Scripts for implementing transfer learning using pre-trained models.
  - `transfer_learning.py`: Implements transfer learning techniques for different datasets.

- **utils/**: Utility scripts used throughout the project.
  - `eval.py`: Code for model evaluation.
  - `metrics.py`: Custom metrics for model performance evaluation.
  - `preprocessing.py`: Code for preprocessing the input data before feeding it to the models.
  - `train.py`: Script for training the models.
  - `.gitignore`: Ignores unnecessary files in the utils folder.

- **README.md**: This file, providing an overview of the repository structure and instructions on how to use the code.

- **requirements.txt**: Lists all the required Python libraries and dependencies for running the project.

## How to Use

1. Clone the repository:
  ```bash
  git clone https://github.com/yourusername/model-compression.git
  cd model-compression
  ```

2. Install the dependencies:
  ```bashbash
  pip install -r requirements.txt
  ```
3. Explore the Jupyter notebooks in the `notebooks/` directory to get started with training and evaluation on your own datasets.

4. For transfer learning, refer to the `transfer_learning.py` script in the `transfer_learning/` folder.

5. Use `main.py` for the overall execution of model training and compression.
