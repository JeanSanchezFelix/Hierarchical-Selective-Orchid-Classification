import os
import logging
from typing import Optional, Dict, Any, List, Union

import numpy as np

from model_compression.src.utils.metrics import (
    calculate_metrics,
    plot_metric_bar,
    plot_confusion_matrix,
    plot_roc_auc_curve,
    plot_radar_chart,
    plot_calibration_curve,
    plot_log_loss
)

try:
    import tensorflow as tf
    from tensorflow import Tensor
    from tensorflow.data import Dataset as TFDataset
except ImportError:
    tf = None

### TODO: Test if this works ###

def test_inference_savedmodel(
    saved_model_path: str,
    test_dir: str,
    batch_size: int = 32,
    image_size: Union[tuple, List[int]] = (224, 224),
    save_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate a TensorFlow SavedModel on a test dataset directory.

    Args:
        saved_model_path: Path to the SavedModel directory.
        test_dir: Directory containing a 'test' subfolder with class subdirectories.
        batch_size: Batch size for evaluation.
        image_size: Tuple specifying (height, width) to resize images.
        save_dir: Optional directory to save evaluation plots.

    Returns:
        A dict of evaluation metrics ('accuracy', 'precision', 'recall', 'f1_score', 'auc', etc.).

    Raises:
        ImportError: If TensorFlow is not installed.
        FileNotFoundError: If model directory or test_dir/test does not exist.
    """
    if tf is None:
        raise ImportError("TensorFlow is required for SavedModel evaluation but is not installed.")
    if not os.path.isdir(saved_model_path):
        raise FileNotFoundError(f"SavedModel directory not found: {saved_model_path}")

    # Load SavedModel
    model = tf.saved_model.load(saved_model_path)
    infer = model.signatures.get("serving_default")
    if infer is None:
        raise ValueError("SavedModel does not have a 'serving_default' signature.")

    # Prepare test dataset
    full_test_dir = os.path.join(test_dir, 'test')
    if not os.path.isdir(full_test_dir):
        raise FileNotFoundError(f"Test directory not found: {full_test_dir}")

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        full_test_dir,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )
    class_names = dataset.class_names

    # Normalization and channel format conversion
    def preprocess(batch_x, batch_y):
        # Rescale to [0,1]
        x = tf.image.convert_image_dtype(batch_x, tf.float32)
        # Transpose HWC to CHW
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        # Normalize using ImageNet stats
        mean = tf.constant([0.485, 0.456, 0.406], shape=[1,3,1,1])
        std = tf.constant([0.229, 0.224, 0.225], shape=[1,3,1,1])
        x = (x - mean) / std
        return x, batch_y

    dataset = dataset.map(preprocess)

    y_true: List[int] = []
    y_pred: List[int] = []
    y_proba: List[List[float]] = []

    # Inference loop
    for x_batch, y_batch in dataset:
        inputs = {list(infer.structured_input_signature[1].keys())[0]: x_batch}
        outputs = infer(**inputs)
        # Get first output tensor
        out_tensor = list(outputs.values())[0]
        logits = out_tensor.numpy()

        # Predictions
        if logits.ndim == 1 or logits.shape[1] == 1:
            probs = 1 / (1 + np.exp(-logits))
            preds = (probs >= 0.5).astype(int).flatten()
        else:
            exp = np.exp(logits)
            probs = exp / np.sum(exp, axis=1, keepdims=True)
            preds = np.argmax(probs, axis=1)

        y_true.extend(y_batch.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_proba.extend(probs.tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_proba_arr = np.array(y_proba)

    metrics = calculate_metrics(y_true_arr, y_pred_arr, y_proba_arr)
    metrics['accuracy'] = float((y_pred_arr == y_true_arr).mean())
    logging.info(f"SavedModel evaluation metrics: {metrics}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_metric_bar(metrics, title="TF SavedModel Metrics", save_path=os.path.join(save_dir, "tf_metrics.png"))
        plot_confusion_matrix(y_true_arr, y_pred_arr, labels=class_names, save_path=os.path.join(save_dir, "tf_confusion.png"))
        plot_radar_chart(metrics, save_path=os.path.join(save_dir, "tf_radar_chart.png"))
        plot_log_loss(metrics, title="Log Loss over epochs", save_path=os.path.join(save_dir, "tf_log_loss.png"))
        plot_roc_auc_curve(y_true_arr, y_proba_arr, save_path=os.path.join(save_dir, "tf_roc_auc.png"))
        plot_calibration_curve(y_true_arr, y_proba_arr, save_path=os.path.join(save_dir, "tf_calibration.png"))
        logging.info(f"Saved TF SavedModel plots to {save_dir}")

    return metrics
