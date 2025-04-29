import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_compression.src.eval.predictions import _compute_predictions
from model_compression.src.utils.metrics import calculate_metrics, plot_metric_bar, plot_confusion_matrix, plot_roc_auc_curve, plot_radar_chart

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

def test_inference_onnx(
    onnx_model_path: str,
    test_loader: DataLoader,
    device: torch.device,  # Mostly will be CPU for ONNX
    save_dir: str = None
) -> None:
    """
    Performs inference on the test dataset using an ONNX model, computes metrics, and optionally saves plots.
    
    This function uses ONNX Runtime to run inference on the exported ONNX model.
    
    Args:
        onnx_model_path (str): Path to the exported ONNX model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to use for inference (typically CPU for ONNX).
        save_dir (str): Directory to save plots. If None, plots will not be saved.
    """
    # Load the ONNX model with ONNX Runtime.
    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    
    predictions, ground_truths, probabilities = [], [], []
    class_labels = test_loader.dataset.classes

    for inputs, labels in tqdm(test_loader, desc="ONNX Inference Progress"):
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
        ground_truths.extend(labels.cpu().numpy())
        predictions.extend(preds.cpu().numpy())
        probabilities.extend(probs.cpu().numpy())

    # Compute metrics similar to the original function.
    y_true = np.array(ground_truths)
    y_pred = np.array(predictions)
    y_proba = np.array(probabilities) if probabilities else None
    metrics = calculate_metrics(y_true, y_pred, y_proba)

    print("ONNX Metrics Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_metric_bar(metrics, save_path=os.path.join(save_dir, "onnx_metrics_bar_chart.png"))
        if class_labels is not None:
            plot_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                labels=class_labels,
                save_path=os.path.join(save_dir, "onnx_confusion_matrix.png")
            )
        if y_proba is not None:
            plot_roc_auc_curve(y_true, y_proba, save_path=os.path.join(save_dir, "onnx_roc_auc_curve.png"))
        plot_radar_chart(metrics, save_path=os.path.join(save_dir, "onnx_radar_chart.png"))
        logging.info(f"ONNX plots saved to {save_dir}")