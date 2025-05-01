import os
import logging
from typing import Optional, Dict, Any, List, Union

import numpy as np
from tqdm import tqdm

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

def test_inference_tflite(
    tflite_model_path: str,
    test_dataset: 'TFDataset',
    input_type: Any = None,
    save_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Run inference with a TFLite model and compute evaluation metrics.

    Args:
        tflite_model_path: Path to the .tflite file.
        test_dataset: A tf.data.Dataset yielding (input, label).
        input_type: Optional TF dtype for inputs (e.g., tf.uint8). If None, infer from interpreter.
        save_dir: Directory to save plots. If None, no plots are saved.

    Returns:
        A dict of metrics including 'accuracy' and, if probabilities available, ROC-AUC.

    Raises:
        ImportError: If TensorFlow is not installed.
        FileNotFoundError: If the TFLite model file does not exist.
    """
    if tf is None:
        raise ImportError("TensorFlow is required for TFLite evaluation but is not installed.")
    if not os.path.isfile(tflite_model_path):
        raise FileNotFoundError(f"TFLite model not found: {tflite_model_path}")

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_true: List[int] = []
    y_pred: List[int] = []
    y_proba: List[List[float]] = []

    for batch in test_dataset:
        inputs, labels = batch
        array = inputs.numpy() if hasattr(inputs, 'numpy') else np.array(inputs)
        # Cast and reshape
        dtype = input_type or input_details[0]['dtype']
        array = array.astype(dtype.as_numpy_dtype if hasattr(dtype, 'as_numpy_dtype') else dtype)
        if array.ndim == len(input_details[0]['shape']) - 1:
            array = np.expand_dims(array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        preds = np.argmax(output, axis=-1).flatten().tolist()
        y_pred.extend(preds)
        y_true.extend([int(label.numpy()) if hasattr(label, 'numpy') else int(label) for label in labels])
        # store probability distribution
        y_proba.extend(output.tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_proba_arr = np.array(y_proba) if y_proba else None

    metrics = calculate_metrics(y_true_arr, y_pred_arr, y_proba_arr)
    # accuracy is always present
    metrics['accuracy'] = float((y_pred_arr == y_true_arr).mean())

    logging.info(f"TFLite evaluation metrics: {metrics}")

    # Save plots if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_metric_bar(metrics, title="Test Metrics", save_path=os.path.join(save_dir, "tflite_metrics.png"))
        plot_confusion_matrix(
            y_true_arr, y_pred_arr, labels=None,
            title="Confusion Matrix", save_path=os.path.join(save_dir, "tflite_confusion_matrix.png")
        )
        plot_radar_chart(metrics, save_path=os.path.join(save_dir, "tflite_radar_chart.png"))
        plot_log_loss(metrics, title="Log Loss over epochs", save_path=os.path.join(save_dir, "tflite_log_loss.png"))
        if y_proba_arr is not None:
            plot_roc_auc_curve(
                y_true_arr, y_proba_arr,
                title="ROC-AUC Curve", save_path=os.path.join(save_dir, "tflite_roc_auc.png")
            )
            plot_calibration_curve(
                y_true_arr, y_proba_arr,
                title="Calibration Curve", save_path=os.path.join(save_dir, "tflite_calibration.png")
            )
        logging.info(f"Saved TFLite evaluation plots to {save_dir}")

    return metrics