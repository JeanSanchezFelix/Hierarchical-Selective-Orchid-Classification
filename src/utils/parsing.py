import os
import csv
import yaml
import json
import argparse
import logging
from utils.callbacks.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Registry mapping callback names to constructors
CALLBACK_REGISTRY = {
    "EarlyStopping": EarlyStopping,
    "ReduceLROnPlateau": ReduceLROnPlateau,
    "ModelCheckpoint": ModelCheckpoint,
}

# Configure logging
def configure_logging(logs: bool, save_dir: str):
    """
    Configures logging to write to a file and optionally to the console.

    Parameters:
        logs (bool): Whether to log messages to the console.
        save_dir (str): Directory where the log file will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, 'training.log')

    # Create or retrieve the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler for writing logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Optional stream handler for logging to the console
    if logs:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(stream_handler)

def process_callbacks(callback_names: list, save_dir: str) -> list[Callback]:
    """
    Processes a list of callback names into instantiated Callback objects.

    Parameters:
        callback_names (list): List of callback names as strings.
        save_dir (str): Directory to save model checkpoints.

    Returns:
        list: List of instantiated Callback objects.
    """
    callbacks = []
    for name in callback_names:
        if name not in CALLBACK_REGISTRY:
            raise ValueError(f"Callback '{name}' is not recognized. Available callbacks: {list(CALLBACK_REGISTRY.keys())}")
        
        # Instantiate callback with default or user-specified parameters
        if name == "EarlyStopping":
            callbacks.append(CALLBACK_REGISTRY[name](monitor="val_loss", patience=3, mode="min", save_path=f"{save_dir}/best_model.pth", verbose=False))
        elif name == "ReduceLROnPlateau":
            callbacks.append(CALLBACK_REGISTRY[name](monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=False))
        elif name == "ModelCheckpoint":
            callbacks.append(CALLBACK_REGISTRY[name](monitor="val_loss", save_best_only=True, mode="min", filepath=f"{save_dir}/best_model.pth", verbose=False))

    return callbacks

# Function to load arguments from a config file (.csv)
def load_args_from_file(file_path):
    """
    Reads arguments from a configuration file (CSV, JSON, YAML) and returns them as a dictionary.
    """
    args_from_file = {}
    try:
        if file_path.endswith(".csv"):
            with open(file_path, mode="r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key, value in row.items():
                        args_from_file[key] = value.strip() if value else None
        elif file_path.endswith(".json"):
            with open(file_path, mode="r") as f:
                args_from_file = json.load(f)
        elif file_path.endswith(".yaml") or file_path.endswith(".yml"):
            with open(file_path, mode="r") as f:
                args_from_file = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv, .json, or .yaml file.")
    except Exception as e:
        raise ValueError(f"Error reading the config file: {e}")
    return args_from_file

# TODO: Simplify Command line arguments parsing? Force some of the mto be from conifguration files
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
            - LOGGING (int): Configure if logs will be displayed on the CLI.

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
    parser.add_argument('--data_augmentation', type=str, choices=["0", "1"], help='Enable data augmentation: 0 (False), 1 (True)')
    parser.add_argument('--callbacks', type=str, nargs='*', choices=["EarlyStopping", "ModelCheckpoint", "ReduceLROnPlateau"],
                        help='Callbacks used in training (EarlyStopping, ModelCheckpoint, etc.)')
    parser.add_argument('--pretrained_weights', type=str, help='Path to pretrained weights')
    parser.add_argument('--save_dir', type=str, help='Directory to save the models')
    parser.add_argument('--logging', type=str, choices=["0", "1"], help='Enable CLI logging: 0 (False), 1 (True)')

    # Parse command-line arguments
    args = parser.parse_args()

    # Load configuration from file if provided
    config_args = load_args_from_file(args.config_file) if args.config_file else {}

    # Set default or override config file arguments with command-line arguments
    #TODO: Error when CLI arguments are int (0,1) since they are treated as False or True
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
    CALLBACKS = get_arg_value('callbacks', ['ModelCheckpoint'])
    PRETRAINED_WEIGHTS = get_arg_value('pretrained_weights', None)
    SAVE_DIR = get_arg_value('save_dir', f'models')
    LOGGING = bool(int(get_arg_value('logging', 1)))

    # Configure logging
    configure_logging(LOGGING, SAVE_DIR)

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
    logging.info(f"CALLBACKS: {CALLBACKS}")
    logging.info(f"PRETRAINED_WEIGHTS: {PRETRAINED_WEIGHTS}")
    logging.info(f"SAVE_DIR: {SAVE_DIR}")

    # Initialize Callbacks
    CALLBACKS = process_callbacks(CALLBACKS, SAVE_DIR)

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
        "CALLBACKS": CALLBACKS,
        "PRETRAINED_WEIGHTS": PRETRAINED_WEIGHTS,
        "SAVE_DIR": SAVE_DIR
    }

# if __name__ == "__main__":
