import os
import logging
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

from model_compression.src.eval.predictions import _compute_predictions
from model_compression.src.utils.metrics import (
    calculate_metrics,
    plot_metric_bar,
    plot_confusion_matrix,
    plot_roc_auc_curve,
    plot_radar_chart,
    plot_calibration_curve,
    plot_log_loss
)


def test_inference_onnx(
    onnx_model_path: str,
    data_loader: DataLoader,
    save_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Run inference using an ONNX model and compute evaluation metrics.

    Args:
        onnx_model_path: Path to the ONNX model file.
        data_loader: DataLoader for test data.
        save_dir: Optional directory to save plots.

    Returns:
        A dict of evaluation metrics.

    Raises:
        ImportError: If onnxruntime is not installed.
        FileNotFoundError: If the ONNX model file is not found.
    """
    # Initialize ONNX Runtime session
    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    
    y_true: List[int] = []
    y_pred: List[int] = []
    y_proba: List[List[float]] = []

    # Retrieve class labels from dataset
    classes = getattr(data_loader.dataset, 'classes', None)

    for inputs, labels in tqdm(data_loader, desc="ONNX Inference Progress"):
        # Convert input to NumPy array (and ensure it's on CPU).
        np_inputs = inputs.cpu().numpy()
        # Run inference using ONNX Runtime.
        ort_outs = session.run(None, {input_name: np_inputs})
        # Assume the first output is the prediction.
        outputs = ort_outs[0]
        # If needed, convert outputs to torch tensors for further processing.
        outputs_tensor = torch.tensor(outputs)
        # Compute predictions (this function must work with tensor or can be modified accordingly).
        probs, preds = _compute_predictions(outputs_tensor)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_proba.extend(probs.cpu().numpy())

    # Compute metrics similar to the original function.
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_proba_arr = np.array(y_proba) if y_proba else None
    metrics = calculate_metrics(y_true_arr, y_pred_arr, y_proba_arr)
    logging.info(f"ONNX inference metrics: {metrics}")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save plots if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_metric_bar(metrics, title="ONNX Metrics", save_path=os.path.join(save_dir, "ONNX_metrics.png"))
        plot_confusion_matrix(
            y_true_arr, y_pred_arr, labels=classes,
            title="Confusion Matrix", save_path=os.path.join(save_dir, "ONNX_confusion_matrix.png")
        )
        plot_radar_chart(metrics, save_path=os.path.join(save_dir, "ONNX_radar_chart.png"))
        # plot_log_loss(metrics, title="Log Loss over epochs", save_path=os.path.join(save_dir, "ONNX_log_loss.png"))
        if y_proba_arr is not None:
            plot_roc_auc_curve(
                y_true_arr, y_proba_arr,
                title="ROC-AUC Curve", save_path=os.path.join(save_dir, "ONNX_roc_auc.png")
            )
            plot_calibration_curve(
                y_true_arr, y_proba_arr,
                title="Calibration Curve", save_path=os.path.join(save_dir, "ONNX_calibration.png")
            )
        logging.info(f"Saved ONNX evaluation plots to {save_dir}")
    
    return metrics