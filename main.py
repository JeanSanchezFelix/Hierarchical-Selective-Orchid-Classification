import logging
from src.utils import parse, load_data, evaluate
from src.train import train_models 

# Add the root directory to sys.path to access datasets
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# # Add the src directory to sys.path to access utils
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    """
    Main function for parsing, data preprocessing, and model training.
    """
    # Step 1: Parse the command-line arguments or configuration file
    logging.info("Parsing arguments...")
    args = parse()

    # Step 2: Load and preprocess the dataset
    logging.info("Loading and preprocessing data...")
    data_loaders = load_data(
        dataset=args['DATASET'],
        batch_size=args['BATCH_SIZE'],
        train_split=args['TRAIN_SPLIT'],  
        test_split=args['TEST_SPLIT'],
        img_size=args['IMG_SIZE'],  
        use_augmentation=args['DATA_AUGMENTATION']
    )

    # Step 3: Train the models
    logging.info("Starting model training...")
    train_models(
        model_name=args['MODEL_NAME'],
        data_loaders=data_loaders,
        save_dir=args['SAVE_DIR'],
        learning_rate=args['LEARNING_RATE'],
        epochs=args['EPOCHS'],
        optimizer=args['OPTIMIZER'],
        criterion=args['CRITERION'],
        callbacks=args['CALLBACKS'],
        pretrained_weights=args['PRETRAINED_WEIGHTS']
    )

    # Step 4: Evaluate models

    evaluate(f"{args['SAVE_DIR']}/{args['DATASET']}/{args['MODEL_NAME']}_best_model.pth", 
        args['IMG_SIZE'], args['DATASET'], 
        f"{args['SAVE_DIR']}/{args['DATASET']}"
    )


    logging.info("Training completed successfully.")

if __name__ == "__main__":
    main()
