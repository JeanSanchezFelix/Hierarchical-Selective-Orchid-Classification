import os
import logging
import argparse
import torch

from datasets.registry import DATASET_REGISTRY
from model_compression.src.utils import configure_logging
from model_compression.src.utils import load_data, process_callbacks
from model_compression.src.train import train_kd 

def parse() -> dict[str, int | str | list]:

    # Argument Parsing
    parser = argparse.ArgumentParser(
        description="Configure and launch model training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and data group
    model_group = parser.add_argument_group("Model Parameters")
    model_group.add_argument("--teacher_name", type=str, default="mobilenet_v2", help="Name of the pre-trained model.")
    model_group.add_argument("--student_name", type=str, default="mobilenet_v2", help="Name of the pre-trained model.")
    model_group.add_argument('--teacher_model_weights', type=str, default=None, help='Path to pretrained teacher weights.')
    model_group.add_argument('--dataset', type=str, choices=list(DATASET_REGISTRY.keys()), required=True,
                            help='Dataset that will be used for knowledge distillation.')
    
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
    misc_group.add_argument('--logging', action="store_true", help='Enable CLI logging: 0 (False), 1 (True)')

    # Parse args
    config = vars(parser.parse_args())  # type: ignore

    # Adjust pre-trained teacher weights
    config['teacher_model_weights'] = os.path.join(config['save_dir'], config['dataset'], f"{config['teacher_name']}_best_model.pth")

    # Configure logging
    configure_logging(enable_console=config['logging'], log_dir=config['save_dir'])

    callback_instances = process_callbacks(config)

    return {
        'TEACHER_NAME': config['teacher_name'],
        'STUDENT_NAME': config['student_name'],
        'TEACHER_MODEL_WEIGHTS': config['teacher_model_weights'],
        'DATASET': config['dataset'],
        'EPOCHS': config['epochs'],
        'BATCH_SIZE': config['batch_size'],
        'LEARNING_RATE': config['learning_rate'],
        'CRITERION': config['criterion'],
        'OPTIMIZER': config['optimizer'],
        'CALLBACKS': list(callback_instances.values()),
        'SAVE_DIR': config['save_dir'],
        'SAVE_NAME': config['save_name']
    }

def main():
    """
    Main function for parsing, data preprocessing, and model training.
    """
    # Step 1: Parse the command-line arguments or configuration file
    logging.info("Parsing arguments...")
    args = parse()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 2: Load and preprocess the dataset
    logging.info("Loading and preprocessing data...")
    data_loaders = load_data(
        dataset_name=args['DATASET'],
        batch_size=args['BATCH_SIZE'],
        train_split=args['TRAIN_SPLIT'],  
        test_split=args['TEST_SPLIT'],
        img_size=args['IMG_SIZE'],  
        use_augmentation=args['DATA_AUGMENTATION'],
        use_sampler=args['SAMPLER']
    )

    # Step 3: Train a student model with knowledge distillation
    logging.info("Training student model with knowledge distillation...")
    teacher, student = train_kd(
                            teacher_name=args['TEACHER_NAME'], 
                            student_name=args['STUDENT_NAME'],
                            data_loaders=data_loaders,
                            save_dir=args['SAVE_DIR'],
                            learning_rate=args['LEARNING_RATE'],
                            num_epochs=args['EPOCHS'],
                            criterion_name = args['CRITERION'],
                            optimizer_name = args['OPTIMIZER'],
                            callbacks=args['CALLBACKS'],
                            quant_mode=None,
                            teacher_weights_path=args['TEACHER_MODEL_WEIGHTS'],
                            device=device,
                    )
    
if __name__ == "__main__":
    main()