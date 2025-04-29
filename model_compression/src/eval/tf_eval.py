import os
import logging
import numpy as np
from tqdm import tqdm
from model_compression.src.utils.metrics import calculate_metrics, plot_metric_bar, plot_confusion_matrix, plot_roc_auc_curve, plot_radar_chart

try:
    import tensorflow as tf
except ImportError:
    tf = None

def test_inference_savedmodel(
    saved_model_path: str,
    data_dir: str,
    batch_size: int = 32,
    image_size: tuple = (224, 224),
    save_dir: str = None
) -> None:
    """
    Performs inference on the test dataset using a TensorFlow SavedModel,
    computes metrics, and optionally saves plots.

    The test dataset is loaded from a directory structure containing "train", "val", and "test" folders.
    It assumes that the test data is in the "test" folder within data_dir.

    Args:
        saved_model_path (str): Path to the TensorFlow SavedModel directory.
        data_dir (str): Root directory containing "train", "val", and "test" subdirectories.
        batch_size (int): Batch size for the test dataset. Defaults to 32.
        image_size (tuple): Desired image size (height, width). Defaults to (224, 224).
        save_dir (str): Directory to save plots. If None, plots will not be saved.
    """
    logging.info("Loading SavedModel for inference...")
    # Load the SavedModel; note that this does not return a Keras model with predict()
    model = tf.saved_model.load(saved_model_path)
    
    # Get the serving signature; this is a callable that accepts a dict of inputs.
    infer = model.signatures["serving_default"]
    
    # Determine the input key from the signature.
    input_key = list(infer.structured_input_signature[1].keys())[0]
    logging.info(f"Model input key: {input_key}")
    
    # Load the test dataset from the "test" folder.
    test_dir = os.path.join(data_dir, "test")
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False  # For reproducible order.
    )
    class_labels = test_dataset.class_names
    logging.info(f"Found classes: {class_labels}")
    
    # Preprocessing: 
    # 1. Rescale images from [0, 255] to [0, 1].
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    # 2. Transpose images from channels-last (H, W, C) to channels-first (C, H, W).
    # 3. Normalize using the same mean and std as during training.
    test_dataset = test_dataset.map(lambda x, y: (
        normalize_channels_first(tf.transpose(normalization_layer(x), perm=[0, 3, 1, 2])),
        y
    ))
    
    predictions_list = []
    ground_truths_list = []
    
    logging.info("Running inference on the test dataset using the serving signature...")
    # Iterate over the dataset and call the signature on each batch.
    for inputs, labels in tqdm(test_dataset, desc="TF Inference Progress"):
    # for inputs, labels in test_dataset:
        # Call the model's serving signature with the proper input key.
        outputs = infer(**{input_key: inputs})
        # Assume the first output is the one we need.
        output_key = list(outputs.keys())[0]
        batch_predictions = outputs[output_key]
        predictions_list.append(batch_predictions.numpy())
        ground_truths_list.append(labels.numpy())
    
    # Concatenate all predictions and ground truths.
    predictions = np.concatenate(predictions_list, axis=0)
    ground_truths = np.concatenate(ground_truths_list, axis=0)

    # Process predictions depending on the number of classes.
    if len(class_labels) == 2:
        # For binary classification: apply sigmoid and threshold at 0.5.
        probs = tf.nn.sigmoid(predictions).numpy()
        pred_labels = (probs >= 0.5).astype(int).flatten()
    else:
        # For multi-class classification: apply softmax and use argmax.
        probs = tf.nn.softmax(predictions, axis=-1).numpy()
        pred_labels = np.argmax(probs, axis=-1)
    
    # Calculate metrics using your helper function.
    metrics: dict[str, float] = calculate_metrics(ground_truths, pred_labels, probs)
    
    # Print the metrics.
    print("Metrics Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Generate and save plots if a save directory is provided.
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_metric_bar(metrics, save_path=os.path.join(save_dir, "metrics_bar_chart.png"))
        if class_labels:
            plot_confusion_matrix(
                y_true=ground_truths,
                y_pred=pred_labels,
                labels=class_labels,
                save_path=os.path.join(save_dir, "confusion_matrix.png")
            )
        if probs is not None:
            plot_roc_auc_curve(ground_truths, probs, save_path=os.path.join(save_dir, "roc_auc_curve.png"))
        plot_radar_chart(metrics, save_path=os.path.join(save_dir, "radar_chart.png"))
        logging.info(f"Plots saved to {save_dir}")

def normalize_channels_first(x: tf.Tensor) -> tf.Tensor:
    """
    Normalizes a tensor in channels-first format using the ImageNet mean and std.
    
    Args:
        x (tf.Tensor): Input tensor of shape (batch, 3, height, width) with values in [0, 1].
    
    Returns:
        tf.Tensor: Normalized tensor.
    """
    # Define the mean and std as constants.
    mean = tf.constant([0.485, 0.456, 0.406], shape=(3, 1, 1), dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], shape=(3, 1, 1), dtype=tf.float32)
    return (x - mean) / std