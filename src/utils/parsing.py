import os
import csv
import yaml
import json
import argparse
import logging
from src.utils.callbacks import process_callbacks

# Function to load arguments from a config file (.csv)
def load_args_from_file(file_path: str) -> dict[str,str]:
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

def validate_args(args):
    """
    Validate the parsed arguments.

    Parameters:
        args (dict): Parsed arguments.

    Raises:
        ValueError: If validation checks fail.
    """
    if args["img_size"] <= 0:
        raise ValueError("The image size (--img_size) must be a positive integer.")
    if args["epochs"] <= 0:
        raise ValueError("The number of epochs (--epochs) must be a positive integer.")
    if args["batch_size"] <= 0:
        raise ValueError("The batch size (--batch_size) must be a positive integer.")
    if args["learning_rate"] <= 0:
        raise ValueError("The learning rate (--learning_rate) must be a positive number.")
    if not (0.0 < args["train_split"] <= 1.0):
        raise ValueError("The training split (--train_split) must be between 0.0 and 1.0.")
    if not (0.0 <= args["test_split"] < 1.0):
        raise ValueError("The test split (--test_split) must be between 0.0 and 1.0.")
    if args["train_split"] + args["test_split"] >= 1.0:
        raise ValueError("The sum of train_split and test_split must be less than 1.0.")
    if args["pretrained_weights"] and not os.path.exists(args["pretrained_weights"]):
        raise ValueError(f"The specified pretrained weights ({args["pretrained_weights"]:}) do not exist.")

    # Logging configuration details
    logging.info("Configuration:")

    for argument, value in args.items():
        logging.info(f"{argument.upper()}: {value}")


# TODO: Simplify Command line arguments parsing? Force some of the parameters to be from conifguration files?
def parse() -> dict[str, int | str | list]:
    """
    Parse command-line arguments and configuration file inputs for training models.

    This function handles the parsing of various configuration parameters needed for
    training models. It supports command-line arguments and optionally loads additional
    configurations from a CSV file. Command-line arguments override the configurations
    loaded from the file.

    Returns:
        dict: A dictionary containing the parsed and validated configuration values:
            - MODEL_NAME (str): Name of the pre-trained model to use.
            - EPOCHS (int): Number of epochs for training.
            - DATASET (str): Dataset that will be used for training.
            - BATCH_SIZE (int): Batch size for training.
            - LEARNING_RATE (float): Learning rate for the optimizer.
            - CRITERION (str): Criterion to use.
            - OPTIMIZER (str): Optimizer to use.
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

    parser = argparse.ArgumentParser(
        description="Train models with configurable parameters and callbacks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model parameters group
    model_group = parser.add_argument_group("Model Parameters")
    model_group.add_argument("--model_name", type=str, default="mobilenet_v2", help="Name of the pre-trained model.")
    model_group.add_argument('--pretrained_weights', type=str, default=None, help='Path to pretrained weights.')
    model_group.add_argument("--img_size", type=int, default=224, help="Image size for model input.")

    # Training parameters group
    training_group = parser.add_argument_group("Training Parameters")
    training_group.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    training_group.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    training_group.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    training_group.add_argument('--criterion', type=str, choices=["cross_entropy", "mse", "l1", "nll", "bce", "bce_with_logits"], 
                        default="cross_entropy", help='Optimizer to use')
    training_group.add_argument("--optimizer", type=str, choices=["adam", "sgd", "rmsprop", "adagrad", "adamw"], 
                                default="adam", help="Optimizer.")
    
    # Callbacks parameters group
    callback_group = parser.add_argument_group("Callback Parameters")
    callback_group.add_argument("--callbacks", type=str, nargs="*", default=["ModelCheckpoint"], 
                                choices=["EarlyStopping", "ModelCheckpoint", "ReduceLROnPlateau"], help="List of callbacks.")

    # ModelCheckpoint parameters
    callback_group.add_argument("--ModelCheckpoint_monitor", type=str, default="val_loss", 
                                 help="Monitor for ModelCheckpoint.")
    callback_group.add_argument("--ModelCheckpoint_save_best_only", action="store_true", default=True, 
                                 help="Save only the best model for ModelCheckpoint.")
    callback_group.add_argument("--ModelCheckpoint_mode", type=str, choices=["min", "max"], default="min", 
                                 help="Mode for ModelCheckpoint.")
    # callback_group.add_argument("--ModelCheckpoint_save_path", type=str, default="./models/best_model.pth", 
    #                              help="Filepath for saving checkpoints.")
    callback_group.add_argument("--ModelCheckpoint_verbose", action="store_true", default=False, 
                                 help="Verbose for EarlyStopping.")

    # EarlyStopping parameters
    callback_group.add_argument("--EarlyStopping_monitor", type=str, default="val_loss", 
                                 help="Monitor for EarlyStopping.")
    callback_group.add_argument("--EarlyStopping_patience", type=int, default=5, 
                                 help="Patience for EarlyStopping.")
    callback_group.add_argument("--EarlyStopping_min_delta", type=float, default=1e-4, 
                                 help="Minimum delta for EarlyStopping.")
    callback_group.add_argument("--EarlyStopping_mode", type=str, choices=["min", "max"], default="min", 
                                 help="Mode for EarlyStopping.")
    # callback_group.add_argument("--EarlyStopping_save_path", type=str, default="./models/best_model.pth", 
    #                              help="Filepath for EarlyStopping model.")
    callback_group.add_argument("--EarlyStopping_verbose", action="store_true", default=False, 
                                 help="Verbose for EarlyStopping.")

    # ReduceLROnPlateau parameters
    callback_group.add_argument("--ReduceLROnPlateau_monitor", type=str, default="val_loss", 
                                 help="Monitor for ReduceLROnPlateau_monitor.")
    callback_group.add_argument("--ReduceLROnPlateau_factor", type=float, default=0.1, 
                                 help="Factor by which to reduce LR for ReduceLROnPlateau.")
    callback_group.add_argument("--ReduceLROnPlateau_patience", type=int, default=10, 
                                 help="Patience for ReduceLROnPlateau.")
    callback_group.add_argument("--ReduceLROnPlateau_min_delta", type=float, default=1e-4, 
                                 help="Minimum delta for ReduceLROnPlateau.")
    callback_group.add_argument("--ReduceLROnPlateau_mode", type=str, choices=["min", "max"], default="min", 
                                 help="Mode for ReduceLROnPlateau.")
    callback_group.add_argument("--ReduceLROnPlateau_min_lr", type=float, default=1e-6, 
                                 help="Minimum learning rate for ReduceLROnPlateau.")
    callback_group.add_argument("--ReduceLROnPlateau_verbose", action="store_true", default=False, 
                                 help="Verbose for ReduceLROnPlateau.")

    # Data parameters group
    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument('--dataset', type=str, choices=["CpAnemia", "MonkeyPox", "SkinCancer"], required=True,
                         help='Dataset that will be used for training.')
    data_group.add_argument("--train_split", type=float, default=0.8, help="Proportion of data for training (0.0-1.0).")
    data_group.add_argument("--test_split", type=float, default=0.1, help="Proportion of data for testing (0.0-1.0).")
    data_group.add_argument("--data_augmentation", action="store_true", default=0, help="Enable data augmentation: 0 (False), 1 (True).")
    data_group.add_argument("--sampler", action="store_true", default=0, help="Enable weighted sampler: 0 (False), 1 (True).")
    data_group.add_argument("--class_weights", action="store_true", default=0, help="Enable class weights: 0 (False), 1 (True).")

    # Miscellaneous parameters group
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument("--save_dir", type=str, default="./models", help="Directory to save models.")
    misc_group.add_argument("--save_name", type=str, default="best_model", help="Name to be used for saving model.")
    misc_group.add_argument("--config_file", type=str, help="Path to a YAML, CSV or JSON configuration file.")
    misc_group.add_argument('--logging', action="store_true", help='Enable CLI logging: 0 (False), 1 (True)')
    

    # Parse command-line arguments
    args = parser.parse_args()

    # Load configuration from file if provided
    config_args = load_args_from_file(args.config_file) if args.config_file else {}

    # Combine arguments (CLI > Config File > Default)
    combined_args = {**config_args, **vars(args)}

    combined_args['save_dir'] = os.path.join(combined_args['save_dir'], combined_args['dataset'])
    
    # Configure logging
    configure_logging(combined_args['logging'], combined_args['save_dir'])

    # Validate arguments
    validate_args(combined_args)

    # Ensure save directory exists
    os.makedirs(combined_args['save_dir'], exist_ok=True)

    # Initialize Callbacks
    CALLBACKS = process_callbacks(combined_args)

    return {
        "MODEL_NAME": combined_args['model_name'],
        "EPOCHS": combined_args['epochs'],
        "DATASET": combined_args['dataset'],
        "BATCH_SIZE": combined_args['batch_size'],
        "LEARNING_RATE": combined_args['learning_rate'],
        "CRITERION": combined_args['criterion'],
        "OPTIMIZER": combined_args['optimizer'],
        "TRAIN_SPLIT": combined_args['train_split'],
        "TEST_SPLIT": combined_args['test_split'],
        "IMG_SIZE": combined_args['img_size'],
        "DATA_AUGMENTATION": combined_args['data_augmentation'],
        "SAMPLER": combined_args['sampler'],
        "CLASS_WEIGHTS": combined_args['class_weights'],
        "CALLBACKS": list(CALLBACKS.values()),
        "PRETRAINED_WEIGHTS": combined_args['pretrained_weights'],
        "SAVE_DIR": combined_args['save_dir']
    }

# if __name__ == "__main__":
