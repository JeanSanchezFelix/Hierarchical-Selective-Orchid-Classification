import logging
from model_compression.src.utils import parse, load_data
from model_compression.src.train import transfer_learning 
from model_compression.src.eval.pytorch_eval import evaluate

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
        dataset_name=args['DATASET'],
        batch_size=args['BATCH_SIZE'],
        train_split=args['TRAIN_SPLIT'],  
        test_split=args['TEST_SPLIT'],
        img_size=args['IMG_SIZE'],  
        use_augmentation=args['DATA_AUGMENTATION'],
        use_sampler=args['SAMPLER']
    )

    # Step 3: Train the models
    logging.info("Starting model training...")
    transfer_learning(
        model_name=args['MODEL_NAME'],
        data_loaders=data_loaders,
        save_dir=args['SAVE_DIR'],
        learning_rate=args['LEARNING_RATE'],
        num_epochs=args['EPOCHS'],
        criterion_name=args['CRITERION'],
        optimizer_name=args['OPTIMIZER'],
        callbacks=args['CALLBACKS'],
        pretrained_weights_path=args['PRETRAINED_WEIGHTS'],
        use_class_weights=args['CLASS_WEIGHTS']
    )

    # Step 4: Evaluate models
    logging.info("Starting model evaluation...")
    # evaluate(
    #     f"{args['SAVE_DIR']}/{args['MODEL_NAME']}_best_model.pth",
    #     f"{args['SAVE_DIR']}/metadata.pth", 
    #     args['IMG_SIZE'], 
    #     args['DATASET'], 
    #     f"{args['SAVE_DIR']}/metrics"
    # )


    logging.info("Training completed successfully.")

if __name__ == "__main__":
    main()
