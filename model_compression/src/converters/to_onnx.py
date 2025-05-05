import os
import logging
import torch
import onnx
import tensorflow as tf

from model_compression.src.quantization.utils.inspect import is_quantized_model

# try:
#     from onnx_tf.backend import prepare
# except ImportError:
#     onnx_tf.backend = None 

def export_pytorch_to_onnx(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    onnx_file_path: str,
    dynamo: bool = True,
    opset_version: int = 18
) -> None:
    """
    Exports a given PyTorch model to the ONNX format.

    This function uses torch.onnx.export to convert the model to ONNX. It assumes the model is in eval mode.
    Note that quantized models may face compatibility issues with ONNX; ensure that the opset_version and model
    configuration are supported.

    Args:
        model: PyTorch model to export (will be set to eval mode).
        example_input: Example input tensor for tracing.
        onnx_file_path: Path to save the ONNX file.
        dynamo: Whether to use Dynamo dynamic shapes API.
        opset_version: ONNX opset version.

    Raises:
        RuntimeError: If torch.onnx.export fails.
    """

    # Set model to evaluation mode.
    if is_quantized_model(model):
        model = torch.ao.quantization.allow_exported_model_train_eval(model)  # restores eval/train to call move_exported_model_* under the hood
    
    model.eval()

    # Prepare the dynamic batch size arguments based on dynamo
    if dynamo:
        dynamic_args = {
            "dynamic_shapes": {
                "input": {0: torch.export.Dim("batch_size")}, 
                "output": {0: torch.export.Dim("batch_size")}
            }
        }
    else:
        dynamic_args = {
            "dynamic_axes": {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        }

    try:
        # Export the model.
        torch.onnx.export(
            model,                          # model being exported
            example_input,                  # example input to the model
            onnx_file_path,                 # where to save the ONNX model
            export_params=True,             # store the trained parameter weights inside the model file
            opset_version=opset_version,    # specify the ONNX version to export the model to
            do_constant_folding=True,       # execute constant folding for optimization
            input_names=['input'],          # the model's input names
            output_names=['output'],        # the model's output names
            dynamo=dynamo,
            external_data=False,
            **dynamic_args                  # unpack the conditional arguments
            # fallback=True,                  # logs when TorchScript fallback occurs
            # report=True                     # generates a markdown report of the export
        )
        logging.info(f"Exported model to ONNX at {onnx_file_path}")
    except Exception as e:
        logging.error(f"ONNX export failed: {e}")
        raise RuntimeError(f"Failed to export to ONNX: {e}") from e

def export_onnx_to_savedmodel(onnx_path: str, saved_model_dir: str) -> None:
    """
    Convert an ONNX model to TensorFlow format and export it as a SavedModel.

    This function loads an ONNX model from the specified path, converts it 
    to TensorFlow format using the ONNX-TF backend, and saves the resulting 
    TensorFlow model as a SavedModel to the given output path.

    Args:
        onnx_path: Path to the ONNX model file.
        saved_model_dir: Output directory for the SavedModel.

    Raises:
        ImportError: If onnx_tf.backend is not available.
        FileNotFoundError: If the ONNX file does not exist.
        RuntimeError: If conversion fails.
    """
    if prepare is None:
        raise ImportError("onnx-tf is required for ONNX to SavedModel conversion but is not installed.")
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    try:
        # Load the ONNX model.
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow representation.
        tf_rep = prepare(onnx_model)
        
        # Export the model as a SavedModel.
        tf_rep.export_graph(saved_model_dir)
        logging.info(f"Saved TensorFlow model at: {saved_model_dir}")
    except Exception as e:
        logging.error(f"ONNX to SavedModel conversion failed: {e}")
        raise RuntimeError(f"Failed to convert ONNX to SavedModel: {e}") from e


# def export_savedmodel_to_tflite(
#     saved_model_dir: str,
#     tflite_model_path: str,
#     calibration_dataloader: torch.utils.data.DataLoader,
#     num_calibration_steps: int = 100
# ) -> None:
#     """
#     Convert a TensorFlow SavedModel to a fully INT8‑quantized TFLite model.

#     Args:
#         saved_model_dir (str): Directory of the TF SavedModel.
#         tflite_model_path (str): Path to output the .tflite file.
#         calibration_dataloader (DataLoader): DataLoader for representative data.
#         num_calibration_steps (int): Number of batches for calibration.
#     """
#     # Create converter
#     converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
#     # Enable default optimizations (including full‑integer quantization)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]  
#     # Provide representative dataset for range calibration
#     converter.representative_dataset = lambda: (
#         next(representative_data_gen(calibration_dataloader))
#         for _ in range(num_calibration_steps)
#     )  
#     # Restrict to INT8 ops only
#     converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  
#     # Ensure integer I/O
#     converter.inference_input_type = tf.int8  
#     converter.inference_output_type = tf.int8 
#     # Convert and save
#     tflite_model = converter.convert()
#     with open(tflite_model_path, "wb") as f:
#         f.write(tflite_model)
#     logging.info(f"Generated TFLite model at: {tflite_model_path}")

def export_savedmodel_to_tflite(
    saved_model_dir: str,
    tflite_model_path: str,
    calibration_dataloader: torch.utils.data.DataLoader,
    num_calibration_steps: int = 100
) -> None:
    """
    Convert a TensorFlow SavedModel to a fully INT8 quantized TFLite model.

    Args:
        saved_model_dir (str): Directory of the TF SavedModel.
        tflite_model_path (str): Path to output the .tflite file.
        calibration_dataloader (DataLoader): DataLoader for representative data.
        num_calibration_steps (int): Number of batches for calibration.
    """
    # Create converter
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    # Enable default optimizations (including full‑integer quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  
    # Provide representative dataset for range calibration
    converter.representative_dataset = lambda: (
        next(representative_data_gen(calibration_dataloader))
        for _ in range(num_calibration_steps)
    )  
    # Restrict to INT8 ops only
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  
    # Ensure integer I/O
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    # Convert and save
    tflite_model = converter.convert()
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"Generated TFLite model at: {tflite_model_path}")