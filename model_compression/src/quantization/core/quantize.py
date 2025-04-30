import os
import torch
import logging
import torch.nn as nn
# import ai_edge_torch
import tensorflow as tf
from typing import Optional
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e, convert_pt2e
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_fx import convert_fx

# from ai_edge_torch.quantize.pt2e_quantizer import get_symmetric_quantization_config
# from ai_edge_torch.quantize.pt2e_quantizer import PT2EQuantizer
# from ai_edge_torch.quantize.quant_config import QuantConfig
# from ai_edge_torch.generative.quantize.quant_recipes import full_int8_dynamic_recipe, full_int8_weight_only_recipe, full_fp16_recipe

from model_compression.src.quantization.utils.calibration import representative_data_gen

def quantize_pytorch_model(model: nn.Module, quant_mode: str, save_dir: Optional[str] = None) -> nn.Module:
    """
    Converts a calibrated/trained PyTorch model to a quantized model using the specified quantization mode.
    
    Args:
        model (nn.Module): The PyTorch model to convert.
        quant_mode (str): The quantization mode to use. Valid options are 'fx' or 'export'.
        save_dir (Optional[str]): Path to save the quantized model's state dictionary. If None, saving is skipped.
    
    Returns:
        nn.Module: The quantized model.
    
    Raises:
        ValueError: If an invalid quantization mode is specified.
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
    if save_dir:
        try:
            torch.save(quantized_model.state_dict(), save_dir)
            size_mb = os.path.getsize(save_dir) / 1e6
            logging.info("Saved quantized mode state dictionary as %s (%.4f MB)", save_dir, size_mb)
        except Exception as e:
            logging.error("Error while saving to %s: %s", save_dir, e)
            raise
    return quantized_model

def post_training_quantization_pytorch_model(model, example_inputs, save_dir: str = "quant_model.tflite"):

    model.eval()

    pt2e_quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
    )

    pt2e_torch_model = capture_pre_autograd_graph(model, example_inputs)
    pt2e_torch_model = prepare_pt2e(pt2e_torch_model, pt2e_quantizer)

    # Run the prepared model with sample input data to ensure that internal observers are populated with correct values
    pt2e_torch_model(*example_inputs)

    # Convert the prepared model to a quantized model
    pt2e_torch_model = convert_pt2e(pt2e_torch_model, fold_quantize=False)

    # Convert to an ai_edge_torch model
    pt2e_drq_model = ai_edge_torch.convert(pt2e_torch_model, example_inputs, quant_config=QuantConfig(pt2e_quantizer=pt2e_quantizer))

    try:
        pt2e_drq_model.export(save_dir)
        size_mb = os.path.getsize(save_dir) / 1e6
        logging.info("Saved post training quantization model as %s (%.4f MB)", save_dir, size_mb)
    except Exception as e:
        logging.error("Error while saving to %s: %s", save_dir, e)
        raise

    return pt2e_drq_model

def post_training_quantization_pytorch_model_tflite(model, example_inputs, recipe: str = "full_int8_dynamic", save_dir: str = "quant_model.tflite"):

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
        
    tfl_model = ai_edge_torch.convert(nhwc_model, nhwc_inputs, quant_config=quant_config)

    try:
        tfl_model.export(save_dir)
        size_mb = os.path.getsize(save_dir) / 1e6
        logging.info("Saved post training quantization model as %s (%.4f MB)", save_dir, size_mb)
    except Exception as e:
        logging.error("Error while saving to %s: %s", save_dir, e)
        raise

    return tfl_model

def post_training_quantization_pytorch_model_tflite_legacy(model, example_inputs, data_loader, save_dir: str = "quant_model.tflite"):

    nhwc_model = ai_edge_torch.to_channel_last_io(model, args=[0]).eval() # Wraps the module with channel first to channel last layout transformations.
    nhwc_inputs = (example_inputs[0].permute(0, 2, 3, 1),)                # Change the inputs to channel last

    rep_ds = lambda: representative_data_gen(data_loader, num_samples=300)

    tfl_converter_flags = {'optimizations': [tf.lite.Optimize.DEFAULT], 
                        'representative_dataset':  rep_ds,
                        'target_spec': {'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]},
                        'inference_input_type': tf.int8,
                        'inference_output_type': tf.int8}

    tfl_int8_model = ai_edge_torch.convert(nhwc_model, nhwc_inputs, _ai_edge_converter_flags=tfl_converter_flags)

    try:
        tfl_int8_model.export(save_dir)
        size_mb = os.path.getsize(save_dir) / 1e6
        logging.info("Saved post training quantization model as %s (%.4f MB)", save_dir, size_mb)
    except Exception as e:
        logging.error("Error while saving to %s: %s", save_dir, e)
        raise

    return tfl_int8_model