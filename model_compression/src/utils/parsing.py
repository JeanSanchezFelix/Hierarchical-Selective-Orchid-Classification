import os
import csv
import json
import yaml
import argparse
import logging
from typing import Any, Dict, Optional, Union, List

from datasets.registry import DATASET_REGISTRY
from model_compression.src.utils.callbacks import process_callbacks
from model_compression.src.utils.logging_setup import configure_logging


def _load_args_from_file(
    file_path: str
) -> Dict[str, Any]:
    """
    Load configuration arguments from a file (CSV, JSON, or YAML).

    Supports:
      - CSV: expects header row with key/value columns.
      - JSON: expects a JSON object.
      - YAML: expects a YAML mapping.

    Args:
        file_path: Path to the configuration file.

    Returns:
        args: A dict of parsed arguments.

    Raises:
        ValueError: For unsupported formats or parsing errors.
    """
    args: Dict[str, Any] = {}
    try:
        ext = os.path.splitext(file_path)[1].lower()
        with open(file_path, 'r') as f:
            if ext == '.csv':
                reader = csv.DictReader(f)
                for row in reader:
                    for k, v in row.items():
                        args[k] = v.strip() if isinstance(v, str) and v.strip() else v
            elif ext == '.json':
                args = json.load(f)
            elif ext in ('.yaml', '.yml'):
                args = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        raise ValueError(f"Failed to load config from {file_path}: {e}")
    return args

def _validate_args(
    args: Dict[str, Any]
) -> None:
    """
    Validate core training configuration arguments.

    Args:
        args: Dictionary of arguments to validate.

    Raises:
        ValueError: If any argument is invalid.
    """
    # Numeric checks
    if args.get('img_size', 0) <= 0:
        raise ValueError("--img_size must be > 0")
    if args.get('epochs', 0) <= 0:
        raise ValueError("--epochs must be > 0")
    if args.get('batch_size', 0) <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.get('learning_rate', 0.0) <= 0.0:
        raise ValueError("--learning_rate must be > 0")

    # Split checks
    train_split = args.get('train_split', 0.0)
    test_split = args.get('test_split', 0.0)
    if not 0.0 < train_split < 1.0:
        raise ValueError("--train_split must be in (0.0, 1.0)")
    if not 0.0 <= test_split < 1.0:
        raise ValueError("--test_split must be in [0.0, 1.0)")
    if train_split + test_split >= 1.0:
        raise ValueError("Sum of train_split and test_split must be < 1.0")

    # Pretrained weights file exists
    pw = args.get('pretrained_weights')
    if pw and not os.path.isfile(pw):
        raise ValueError(f"Pretrained weights not found: {pw}")

    # Log configuration
    logging.info("Training configuration:")
    for k, v in args.items():
        logging.info(f"  {k}: {v}")

def parse() -> dict[str, int | str | list]:
    """
    Parse CLI arguments and optional config file, then validate and return settings.

    Returns:
        A dict containing all training configuration values with keys:
          - MODEL_NAME, PRETRAINED_WEIGHTS, IMG_SIZE, EPOCHS, BATCH_SIZE,
            LEARNING_RATE, CRITERION, OPTIMIZER, CALLBACKS, DATASET,
            TRAIN_SPLIT, TEST_SPLIT, DATA_AUGMENTATION, SAMPLER,
            CLASS_WEIGHTS, SAVE_DIR.

    Raises:
        ValueError: If argument validation fails.
    """
    # Argument Parsing
    parser = argparse.ArgumentParser(
        description="Configure and launch model training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and data group
    model_group = parser.add_argument_group("Model Parameters")
    model_group.add_argument("--model_name", type=str, default="mobilenet_v2", help="Name of the pre-trained model.")
    model_group.add_argument('--pretrained_weights', type=str, default=None, help='Path to pretrained weights.')
    model_group.add_argument("--img_size", type=int, default=224, help="Image size for model input.")
    model_group.add_argument('--dataset', type=str, choices=list(DATASET_REGISTRY.keys()), required=True,
                            help='Dataset that will be used for training.')

    # Training group
    training_group = parser.add_argument_group("Training Parameters")
    training_group.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    training_group.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    training_group.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    training_group.add_argument('--criterion', type=str,
                        choices=["cross_entropy","mse","l1","nll","bce","bce_with_logits"],
                        default='cross_entropy',
                        help="Criterion to use.")
    training_group.add_argument('--optimizer', type=str,
                        choices=["adam","sgd","rmsprop","adagrad","adamw"],
                        default='adam',
                        help="Optimizer to use.")
    
    # Data handling group
    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument("--train_split", type=float, default=0.8, help="Proportion of data for training (0.0-1.0).")
    data_group.add_argument("--test_split", type=float, default=0.1, help="Proportion of data for testing (0.0-1.0).")
    data_group.add_argument("--data_augmentation", action="store_true", default=0, help="Enable data augmentation: 0 (False), 1 (True).")
    data_group.add_argument("--sampler", action="store_true", default=0, help="Enable weighted sampler: 0 (False), 1 (True).")
    data_group.add_argument("--class_weights", action="store_true", default=0, help="Enable class weights: 0 (False), 1 (True).")

    
    # Callbacks group
    callback_group = parser.add_argument_group("Callback Parameters")
    callback_group.add_argument("--callbacks", type=str, nargs="*", default=["ModelCheckpoint"], 
                                choices=["EarlyStopping", "ModelCheckpoint", "ReduceLROnPlateau"], help="List of callbacks.")

    # ModelCheckpoint
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

    # EarlyStopping
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

    # ReduceLROnPlateau
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
    
    # Miscellaneous group
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument("--save_dir", type=str, default="./models", help="Directory to save models.")
    misc_group.add_argument("--save_name", type=str, default="best_model", help="Name to be used for saving model.")
    misc_group.add_argument("--config_file", type=str, default=None, help="Path to a YAML, CSV or JSON configuration file.")
    misc_group.add_argument('--logging', action="store_true", help='Enable CLI logging: 0 (False), 1 (True)')

    # Parse args
    cli_args = vars(parser.parse_args())  # type: ignore

    # Load file overrides
    file_args: Dict[str, Any] = {}
    if cli_args.get('config_file'):
        file_args = _load_args_from_file(cli_args['config_file'])

    # Combine arguments (CLI > Config File > Default)
    config = {**file_args, **cli_args}

    # Adjust save_dir
    config['save_dir'] = os.path.join(config['save_dir'], config['dataset'])
    
    # Configure logging
    configure_logging(enable_console=config['logging'], log_dir=config['save_dir'])

    # Validate
    _validate_args(config)

    # Ensure save directory
    os.makedirs(config['save_dir'], exist_ok=True)

    # Instantiate callbacks
    callback_instances = process_callbacks(config)

    return {
        'MODEL_NAME': config['model_name'],
        'PRETRAINED_WEIGHTS': config['pretrained_weights'],
        'IMG_SIZE': config['img_size'],
        'DATASET': config['dataset'],
        'EPOCHS': config['epochs'],
        'BATCH_SIZE': config['batch_size'],
        'LEARNING_RATE': config['learning_rate'],
        'CRITERION': config['criterion'],
        'OPTIMIZER': config['optimizer'],
        'TRAIN_SPLIT': config['train_split'],
        'TEST_SPLIT': config['test_split'],
        'DATA_AUGMENTATION': config['data_augmentation'],
        'SAMPLER': config['sampler'],
        'CLASS_WEIGHTS': config['class_weights'],
        'CALLBACKS': list(callback_instances.values()),
        'SAVE_DIR': config['save_dir']
    }
