import os
import logging

import torch

try:
    import tflite
except ImportError:
    tflite = None

def check_quantized_modules(model: torch.nn.Module) -> None:
    """
    logging.infos out the module names and their types to help identify if quantized 
    versions of layers are present.
    
    Args:
        model (torch.nn.Module): The quantized model to inspect.
    """
    for name, module in model.named_modules():
        logging.info(f"{name}: {type(module)}")

def is_quantized_model(model: torch.nn.Module) -> bool:
    """
    Checks if the model is quantized based on its type.
    
    The quantized model produced by export quantization is a GraphModule.
    This function returns True if the model's type name matches that.
    
    Args:
        model (torch.nn.Module): The model to check.
    
    Returns:
        bool: True if the model appears quantized, False otherwise.
    """
    return type(model).__name__ == "GraphModule"

def _get_builtin_operator_map() -> dict[int, str]:
    """
    Builds a dictionary mapping numeric operator codes to their string names
    by inspecting tflite.BuiltinOperator.
    
    Returns:
        dict[int, str]: Mapping from op code to op name.
    """
    op_map: dict[int, str] = {}
    for attr_name in dir(tflite.BuiltinOperator):
        if not attr_name.startswith("__"):
            attr_value = getattr(tflite.BuiltinOperator, attr_name)
            if isinstance(attr_value, int):
                op_map[attr_value] = attr_name
    return op_map

def report_tflite_ops(tflite_model_path: str) -> None:
    """
    Reads a TFLite model file and prints a report of the operators in the model.
    It reports the total number of operators, counts for TFLite built-in ops,
    and counts for fallback (SELECT_TF_OPS) ops.
    
    If an operator's built-in code is CUSTOM, this function checks its custom code.
    If the custom code begins with 'Flex', the operator is considered a fallback op.
    
    Args:
        tflite_model_path (str): Path to the .tflite model file.
    """
    if not os.path.exists(tflite_model_path):
        logging.error(f"File {tflite_model_path} not found.")
        return

    with open(tflite_model_path, "rb") as f:
        buf = f.read()

    # Load the FlatBuffer TFLite model.
    model = tflite.Model.GetRootAsModel(buf, 0)
    total_ops = 0
    builtin_ops = 0
    fallback_ops = 0  # Counting SELECT_TF_OPS/fallback ops.
    op_counts: dict[str, int] = {}
    select_tf_ops: set[str] = set()

    # Build our mapping from op code to name for built-in ops.
    op_map = _get_builtin_operator_map()

    # Iterate over each subgraph in the model.
    # A TFLite model can contain multiple subgraphs, each representing a separate computational graph.
    for subgraph_idx in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(subgraph_idx)           # Retrieve the subgraph at index `subgraph_idx`.
        for op_idx in range(subgraph.OperatorsLength()):   # Iterate over all the operators (layers/operations) in the current subgraph.
            op = subgraph.Operators(op_idx)                # Retrieve the operator at index `op_idx`.
            opcode_index = op.OpcodeIndex()                # Get the opcode index, which is an identifier referring to the type of operation.
            op_code = model.OperatorCodes(opcode_index)    # Retrieve the operator code information associated with this operation.
            builtin_code = op_code.BuiltinCode()           # Extract the built-in operation code (an integer that maps to a specific operation type).

            # Check if the operator is a custom op. If so, inspect its custom code.
            if builtin_code == tflite.BuiltinOperator.CUSTOM:
                custom_code = op_code.CustomCode()          # Retrieve the custom operation's string identifier (if available).
                # If the custom code indicates a fallback TensorFlow op, count it as SELECT_TF_OPS.
                # If the custom code starts with "Flex", it indicates a fallback to a TensorFlow op.
                if custom_code is not None and custom_code.startswith(b"Flex"): 
                    op_name = "SELECT_TF_OPS"               # Mark it as a TensorFlow fallback operation.
                    fallback_ops += 1
                    select_tf_ops.add(custom_code)
                else:
                    op_name = "CUSTOM"                      # Otherwise, classify it as a general custom operation.
                    builtin_ops += 1                        # Count it as a built-in op (or modify if treating custom separately).
            else:
                # For non-custom (standard TFLite) operations, look up the operation name using `op_map`.
                op_name = op_map.get(builtin_code, "UNKNOWN")
                builtin_ops += 1

            # Update the operator count dictionary to track occurrences of each type of operation.
            op_counts[op_name] = op_counts.get(op_name, 0) + 1
            total_ops += 1

    print("TFLite Operator Conversion Report:")
    print(f"Total operators: {total_ops}")
    print(f"TFLite Built-in ops: {builtin_ops}")
    print(f"SELECT_TF_OPS (fallback ops): {fallback_ops}")
    print(f"SELECT_TF_OPS (fallback ops): {select_tf_ops}")
    print("\nDetailed operator counts:")
    for op_name, count in op_counts.items():
        print(f"{op_name}: {count}")

def inspect_pth_weights(pth_file: str) -> None:
    """
    Loads a PyTorch model from a .pth file and checks how many tensors are quantized (INT8) vs. non-quantized (FP32).

    Args:
        pth_file (str): Path to the .pth file containing the model weights.
    """
    # Load the model checkpoint
    checkpoint = torch.load(pth_file, map_location="cpu")

    # Initialize counters
    quantized_count = 0
    non_quantized_count = 0
    total_tensors = 0

    for name, tensor in checkpoint.items():
        if isinstance(tensor, torch.Tensor):
            total_tensors += 1  # Count total tensors
            if tensor.is_quantized:
                quantized_count += 1
                print(f"Quantized Tensor: {name}, dtype: {tensor.dtype}, shape: {tensor.shape}, qscheme: {tensor.qscheme()}")
            else:
                non_quantized_count += 1
                print(f"Floating-point Tensor: {name}, dtype: {tensor.dtype}, shape: {tensor.shape}")

    # Print summary
    print("\n===== Summary =====")
    print(f"Total Tensors: {total_tensors}")
    print(f"Quantized (INT8) Tensors: {quantized_count}")
    print(f"Non-Quantized (FP32) Tensors: {non_quantized_count}")
    
    if quantized_count > 0:
        print("\nThe model contains quantized weights.")
    else:
        print("\nNo quantized tensors detected. The model might have been converted back to FP32.")
