import torch
import src.Quantization.utils.conversions.onnx as onnx
from onnx_tf.backend import prepare

def export_pytorch_to_onnx(model: torch.nn.Module, example_input: torch.Tensor, onnx_file_path: str, opset_version: int = 12) -> None:
    """
    Exports a given PyTorch model to the ONNX format.

    This function uses torch.onnx.export to convert the model to ONNX. It assumes the model is in eval mode.
    Note that quantized models may face compatibility issues with ONNX; ensure that the opset_version and model
    configuration are supported.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        example_input (torch.Tensor): An example input tensor with the appropriate shape.
        onnx_file_path (str): Path where the ONNX file will be saved.
        opset_version (int): ONNX opset version to use. Defaults to 12.
    """
    # Ensure the model is in evaluation mode.
    model.eval()

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
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # enable dynamic batch size
        dynamo=False
    )
    print(f"Model successfully exported to {onnx_file_path}")

def export_onnx_to_tf(onnx_path: str, tf_path: str) -> None:
    """
    Convert an ONNX model to TensorFlow format and export it as a SavedModel.

    This function loads an ONNX model from the specified path, converts it 
    to TensorFlow format using the ONNX-TF backend, and saves the resulting 
    TensorFlow model as a SavedModel to the given output path.

    Args:
        onnx_path (str): The file path to the ONNX model.
        tf_path (str): The file path where the converted TensorFlow model 
                       (SavedModel) will be saved.

    Returns:
        None: This function does not return any value. It exports the model to 
              the specified location.
    """
    # Load the ONNX model.
    onnx_model = onnx.load(onnx_path)
    
    # Convert to TensorFlow representation.
    tf_rep = prepare(onnx_model)
    
    # Export the model as a SavedModel.
    tf_rep.export_graph(tf_path)