import os
import csv
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load arguments from a config file (.csv)
def load_args_from_file(file_path):
    """
    Reads arguments from a CSV file and returns them as a dictionary.
    """
    args_from_file = {}
    if file_path.endswith('.csv'):
        try:
            with open(file_path, mode='r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key, value in row.items():
                        args_from_file[key] = value.strip() if value else None
        except Exception as e:
            raise ValueError(f"Error reading the config file: {e}")
    else:
        raise ValueError("Unsupported file format. Please provide a .csv file.")
    return args_from_file

def parse():
    """
    Parse command-line arguments and configuration file inputs for training models.

    This function handles the parsing of various configuration parameters needed for
    training models. It supports command-line arguments and optionally loads additional
    configurations from a CSV file. Command-line arguments override the configurations
    loaded from the file.

    Returns:
        dict: A dictionary containing the parsed and validated configuration values:
            - MODEL_NAME (str): Name of the pre-trained model to use.
            - NUM_MODELS (int): Number of models to train.
            - EPOCHS (int): Number of epochs for training.
            - DATASET_DIR (str): Path to the dataset directory.
            - BATCH_SIZE (int): Batch size for training.
            - LEARNING_RATE (float): Learning rate for the optimizer.
            - OPTIMIZER (str): Optimizer to use ('adam' or 'sgd').
            - TRAIN_SPLIT (float): Proportion of data used for training (0.0 - 1.0).
            - TEST_SPLIT (float): Proportion of data used for testing (0.0 - 1.0).
            - IMG_SIZE (int): Image size for input to the model (e.g., 224 for 224x224 images).
            - DATA_AUGMENTATION (bool): Whether to use data augmentation (True or False).
            - SAVE_DIR (str): Directory where trained models will be saved.

    Raises:
        ValueError: If any of the arguments fail validation checks.
    """
    # Argument Parsing
    parser = argparse.ArgumentParser(description='Train multiple models with custom configurations.')
    parser.add_argument('--config_file', type=str, help='Path to configuration file (CSV)')
    parser.add_argument('--model_name', type=str, help='Name of the pre-trained model to use (mobilenet, efficientnet, etc.)')
    parser.add_argument('--num_models', type=int, help='Number of models to train')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train each model')
    parser.add_argument('--dataset_dir', type=str, help='Directory where the dataset is located')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--criterion', type=str, choices=["cross_entropy", "mse", "l1", "nll", "bce", "bce_with_logits"], 
                        help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer')
    parser.add_argument('--optimizer', type=str, choices=["adam", "sgd", "rmsprop", "adagrad", "adamw"], help='Optimizer to use')
    parser.add_argument('--train_split', type=float, help='Percentage of training split (0.0-1.0)')
    parser.add_argument('--test_split', type=float, help='Percentage of test split (0.0-1.0)')
    parser.add_argument('--img_size', type=int, help='Image size (e.g., 224 for 224x224 images)')
    parser.add_argument('--data_augmentation', type=int, choices=[0, 1], help='Enable data augmentation: 0 (False), 1 (True)')
    parser.add_argument('--pretrained_weights', type=str, help='Path to pretrained weights')
    parser.add_argument('--save_dir', type=str, help='Directory to save the models')

    # Parse command-line arguments
    args = parser.parse_args()

    # Load configuration from file if provided
    config_args = load_args_from_file(args.config_file) if args.config_file else {}

    # Set default or override config file arguments with command-line arguments
    def get_arg_value(arg_name, default):
        """
        Helper function to get argument values, prioritizing CLI input over config file.
        """
        return getattr(args, arg_name) or config_args.get(arg_name, default)

    MODEL_NAME = get_arg_value('model_name', 'mobilenet_v2')
    NUM_MODELS = int(get_arg_value('num_models', 1))
    EPOCHS = int(get_arg_value('epochs', 5))
    DATASET_DIR = get_arg_value('dataset_dir', None)
    BATCH_SIZE = int(get_arg_value('batch_size', 32))
    LEARNING_RATE = float(get_arg_value('learning_rate', 0.001))
    CRITERION = get_arg_value('criterion', 'cross_entropy')
    OPTIMIZER = get_arg_value('optimizer', 'adam')
    TRAIN_SPLIT = float(get_arg_value('train_split', 0.8))
    TEST_SPLIT = float(get_arg_value('test_split', 0.1))
    IMG_SIZE = int(get_arg_value('img_size', 224))
    DATA_AUGMENTATION = bool(int(get_arg_value('data_augmentation', 0)))
    PRETRAINED_WEIGHTS = get_arg_value('pretrained_weights', None)
    SAVE_DIR = get_arg_value('save_dir', f'models')

    # Argument Validation
    if NUM_MODELS <= 0:
        raise ValueError("The number of models (--num_models) must be a positive integer.")
    if EPOCHS <= 0:
        raise ValueError("The number of epochs (--epochs) must be a positive integer.")
    if BATCH_SIZE <= 0:
        raise ValueError("The batch size (--batch_size) must be a positive integer.")
    if LEARNING_RATE <= 0:
        raise ValueError("The learning rate (--learning_rate) must be a positive number.")
    if not DATASET_DIR or not os.path.exists(DATASET_DIR):
        raise ValueError(f"The specified dataset directory ({DATASET_DIR}) does not exist.")
    if not (0.0 < TRAIN_SPLIT <= 1.0):
        raise ValueError("The training split (--train_split) must be between 0.0 and 1.0.")
    if not (0.0 <= TEST_SPLIT < 1.0):
        raise ValueError("The test split (--test_split) must be between 0.0 and 1.0.")
    if TRAIN_SPLIT + TEST_SPLIT >= 1.0:
        raise ValueError("The sum of train_split and test_split must be less than 1.0.")
    if PRETRAINED_WEIGHTS and not os.path.exists(DATASET_DIR):
        raise ValueError(f"The specified pretrained weights ({PRETRAINED_WEIGHTS}) do not exist.")
    if IMG_SIZE <= 0:
        raise ValueError("The image size (--img_size) must be a positive integer.")

    # Ensure save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Logging configuration details
    logging.info("Configuration:")
    logging.info(f"MODEL_NAME: {MODEL_NAME}")
    logging.info(f"NUM_MODELS: {NUM_MODELS}")
    logging.info(f"EPOCHS: {EPOCHS}")
    logging.info(f"DATASET_DIR: {DATASET_DIR}")
    logging.info(f"BATCH_SIZE: {BATCH_SIZE}")
    logging.info(f"LEARNING_RATE: {LEARNING_RATE}")
    logging.info(f"CRITERION: {CRITERION}")
    logging.info(f"OPTIMIZER: {OPTIMIZER}")
    logging.info(f"TRAIN_SPLIT: {TRAIN_SPLIT}")
    logging.info(f"TEST_SPLIT: {TEST_SPLIT}")
    logging.info(f"IMG_SIZE: {IMG_SIZE}")
    logging.info(f"DATA_AUGMENTATION: {DATA_AUGMENTATION}")
    logging.info(f"PRETRAINED_WEIGHTS: {PRETRAINED_WEIGHTS}")
    logging.info(f"SAVE_DIR: {SAVE_DIR}")

    return {
        "MODEL_NAME": MODEL_NAME,
        "NUM_MODELS": NUM_MODELS,
        "EPOCHS": EPOCHS,
        "DATASET_DIR": DATASET_DIR,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "CRITERION": CRITERION,
        "OPTIMIZER": OPTIMIZER,
        "TRAIN_SPLIT": TRAIN_SPLIT,
        "TEST_SPLIT": TEST_SPLIT,
        "IMG_SIZE": IMG_SIZE,
        "DATA_AUGMENTATION": DATA_AUGMENTATION,
        "PRETRAINED_WEIGHTS": PRETRAINED_WEIGHTS,
        "SAVE_DIR": SAVE_DIR
    }

# if __name__ == "__main__":
