import os
import logging
from typing import Optional, Tuple, Generator, List, Any

import torch
import torch.nn as nn
from torch import Tensor

try:
    import ai_edge_torch
except ImportError:
    ai_edge_torch = None
try:
    import tensorflow as tf
except ImportError:
    tf = None

from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantize_fx import convert_fx
from torch._export import capture_pre_autograd_graph

from ai_edge_torch.quantize.pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config
from ai_edge_torch.quantize.quant_config import QuantConfig
from ai_edge_torch.generative.quantize.quant_recipes import (
    full_int8_dynamic_recipe,
    full_int8_weight_only_recipe,
    full_fp16_recipe
)

from model_compression.src.quantization.utils.calibration import representative_data_gen

def quantize_pytorch_model(
    model: nn.Module,
    quant_mode: str,
    save_path: Optional[str] = None
) -> nn.Module:
    """
    Quantize a PyTorch model using FX graph mode or PT2E export mode.

    Args:
        model: A trained PyTorch model.
        quant_mode: 'fx' for FX Graph Mode or 'export' for PT2E Export Mode.
        save_path: Optional path to save the quantized state dict.

    Returns:
        The quantized PyTorch model.

    Raises:
        ValueError: If quant_mode is unsupported.
        IOError: If saving the quantized state dict fails.
    """
    # Convert the model based on the quantization mode.
    if quant_mode == "fx":
        logging.info("Quantizing model using FX Graph Mode.")
        quantized_model = convert_fx(model)
    elif quant_mode == "export":
        logging.info("Quantizing model using PT2E Export Mode.")
        model = torch.ao.quantization.move_exported_model_to_eval(model)
        quantized_model = convert_pt2e(model, fold_quantize=True)
    else:
        raise ValueError("Invalid mode. Choose either 'fx' or 'export'.")
    
    # Save the quantized model's state dictionary if a save directory is provided.
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        try:
            torch.save(quantized_model.state_dict(), save_path)
            size_mb = os.path.getsize(save_path) / 1e6
            logging.info(f"Quantized model saved to {save_path} ({size_mb:.2f} MB)")
        except Exception as e:
            logging.error(f"Failed to save quantized model at {save_path}: {e}")
            raise IOError(f"Could not save quantized state dict to {save_path}") from e
    return quantized_model

def post_training_quantization_pytorch_model(
    model: nn.Module,
    example_inputs: Tuple[Tensor, ...],
    save_path: Optional[str] = None
) -> Any:
    """
    Perform PT2E post-training quantization on a PyTorch model and export via ai_edge_torch.

    Steps:
    1. Capture a static graph.
    2. Prepare with symmetric dynamic quantization config.
    3. Calibrate with example inputs.
    4. Convert to quantized model and export to TFLite using ai_edge_torch.

    Args:
        model: The float PyTorch model.
        example_inputs: Example inputs tuple for tracing and calibration.
        save_path: Path to export the quantized TFLite model.

    Returns:
        The exported ai_edge_torch model instance.

    Raises:
        RuntimeError: If any conversion or export step fails.
    """
    if ai_edge_torch is None:
        raise ImportError('ai_edge_torch is required for PT2E quantization export')

    model.eval()

    # Build quantizer
    config = get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
    pt2e_quantizer = PT2EQuantizer().set_global(config)

    # Capture static graph
    traced = capture_pre_autograd_graph(model, example_inputs)
    pt2e_torch_model = prepare_pt2e(traced, pt2e_quantizer)

    # Run the prepared model with sample input data to ensure that internal observers are populated with correct values (calibrate)
    pt2e_torch_model(*example_inputs)

    # Convert the prepared model to a quantized model
    quantized_model = convert_pt2e(pt2e_torch_model, fold_quantize=False)

    # Convert to an ai_edge_torch model (Export edge model)
    try:
        edge_model = ai_edge_torch.convert(
            quantized_model,
            example_inputs,
            quant_config=QuantConfig(pt2e_quantizer=pt2e_quantizer)
        )
    except Exception as e:
        logging.error(f"ai_edge_torch conversion failed: {e}")
        raise RuntimeError('ai_edge_torch export failed') from e

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        try:
            edge_model.export(save_path)
            size_mb = os.path.getsize(save_path) / 1e6
            logging.info(f"PT2E quantized model exported to {save_path} ({size_mb:.2f} MB)")
        except Exception as e:
            logging.error(f"Failed to export PT2E model: {e}")
            raise RuntimeError(f"Export to {save_path} failed") from e

    return edge_model

def post_training_quantization_pytorch_model_tflite(
    model: nn.Module,
    example_inputs: Tuple[Tensor, ...],
    recipe: str = 'full_int8_dynamic',
    save_path: Optional[str] = None
) -> Any:
    """
    Perform post-training quantization with specified recipe and export to TFLite via ai_edge_torch.

    Args:
        model: The float PyTorch model.
        example_inputs: Example input tuple.
        recipe: Quantization recipe name ('full_int8_dynamic', 'full_int8_weight_only', 'full_fp16').
        save_path: Path to export the quantized TFLite model.

    Returns:
        The exported ai_edge_torch model instance.

    Raises:
        ValueError: If the recipe name is invalid.
        RuntimeError: On conversion or export failure.
    """
    if ai_edge_torch is None:
        raise ImportError('ai_edge_torch is required for TFLite export')

    nhwc_model = ai_edge_torch.to_channel_last_io(model, args=[0]).eval() # Wraps the module with channel first to channel last layout transformations.
    nhwc_inputs = (example_inputs[0].permute(0, 2, 3, 1),)                # Change the inputs to channel last

    # Choose a supported quantization recipe
    if recipe == "full_int8_dynamic":
        quant_config = full_int8_dynamic_recipe()
    elif recipe == "full_int8_weight_only":
        quant_config = full_int8_weight_only_recipe()
    elif recipe == "full_fp16":
        quant_config = full_fp16_recipe()
    else:
        raise ValueError(f"Unsupported recipe: {recipe}. Please choose a valid recipe.")

    # Convert
    try:
        tfl_model = ai_edge_torch.convert(nhwc_model, nhwc_inputs, quant_config=quant_config)
    except Exception as e:
        logging.error(f"TFLite quant conversion failed: {e}")
        raise RuntimeError('TFLite quant conversion failed') from e

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        try:
            tfl_model.export(save_path)
            size_mb = os.path.getsize(save_path) / 1e6
            logging.info(f"TFLite post-training quant model saved to {save_path} ({size_mb:.2f} MB)")
        except Exception as e:
            logging.error(f"Failed to export TFLite model: {e}")
            raise RuntimeError(f"Export to {save_path} failed") from e

    return tfl_model

def post_training_quantization_pytorch_model_tflite_legacy(
    model: nn.Module,
    example_inputs: Tuple[Tensor, ...],
    data_loader: Any,
    save_path: Optional[str] = None
) -> Any:
    """
    Legacy post-training quantization export to TFLite using a representative data generator.

    Args:
        model: The float PyTorch model.
        example_inputs: Example input tuple.
        data_loader: DataLoader to generate calibration samples.
        save_path: Path to export the quantized TFLite model.

    Returns:
        The exported ai_edge_torch model instance.

    Raises:
        ImportError: If ai_edge_torch or tf not installed.
        RuntimeError: On conversion or export failure.
    """
    if ai_edge_torch is None or tf is None:
        raise ImportError('ai_edge_torch and TensorFlow are required for legacy TFLite export')

    nhwc_model = ai_edge_torch.to_channel_last_io(model, args=[0]).eval() # Wraps the module with channel first to channel last layout transformations.
    nhwc_inputs = (example_inputs[0].permute(0, 2, 3, 1),)                # Change the inputs to channel last

    rep_ds = lambda: representative_data_gen(data_loader, num_samples=300)

    tflite_flags = {
        'optimizations': [tf.lite.Optimize.DEFAULT],
        'representative_dataset': rep_ds,
        'target_spec': {'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]},
        'inference_input_type': tf.int8,
        'inference_output_type': tf.int8
    }

    try:
        tfl_int8_model = ai_edge_torch.convert(nhwc_model, nhwc_inputs, _ai_edge_converter_flags=tflite_flags)
    except Exception as e:
        logging.error(f"Legacy TFLite conversion failed: {e}")
        raise RuntimeError('Legacy TFLite quant conversion failed') from e

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        try:
            tfl_int8_model.export(save_path)
            size_mb = os.path.getsize(save_path) / 1e6
            logging.info(f"Legacy TFLite model saved to {save_path} ({size_mb:.2f} MB)")
        except Exception as e:
            logging.error(f"Failed to export legacy TFLite model: {e}")
            raise RuntimeError(f"Export to {save_path} failed") from e

    return tfl_int8_model