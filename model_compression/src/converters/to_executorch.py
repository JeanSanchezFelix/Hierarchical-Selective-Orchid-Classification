import os
import logging
from typing import Optional, Tuple

import torch

from executorch.exir import ExecutorchProgramManager, EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import MemoryPlanningPass
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.extension.export_util.utils import export_to_edge

def save_pte_program(
    prog: ExecutorchProgramManager,
    model_name: str,
    output_dir: str = ""
) -> str:
    """
    Writes an ExecuTorch program to disk as a .pte file.

    Args:
        prog (ExecutorchProgramManager): The compiled ExecuTorch program.
        model_name (str): Base filename (with or without .pte extension).
        output_dir (str): Directory to save into. Defaults to current working dir.

    Returns:
        str: Full path to the written .pte file.
    """
    if model_name.endswith(".pte"):
        filename = model_name
    else:
        filename = os.path.join(output_dir, f"{model_name}.pte")
    try:
        with open(filename, "wb") as f:
            prog.write_to_file(f)
        logging.info("Saved exported program to %s", filename)
    except Exception as e:
        logging.error("Error while saving to %s: %s", filename, e)
        raise
    return filename


def convert_to_executorch_program(model: torch.nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    save_path: Optional[str] = "model.pte",
    verbose: bool = False
) -> str:
    """
    Converts a PyTorch model into an ExecuTorch program and optionally saves it as a .pte file.

    This function performs the following steps:
    1. Exports the PyTorch model to the ATen dialect using Torch Export API.
    2. Applies edge-specific optimizations and lowering via ExecuTorch utilities.
    3. Converts the optimized program into an ExecuTorch "pte" format.
    4. Writes the resulting program buffer to disk if a save path is provided.

    Args:
        model (torch.nn.Module): The trained PyTorch model to convert.
        example_inputs (Tuple[torch.Tensor, ...]): Input tensors for tracing during export.
        save_path (Optional[str]): File path where the .pte ExecuTorch program will be saved. Default is "model.pte".
        verbose (bool): If True, prints intermediate graph representations to stdout.

    Returns:
        ExecuTorchProgram: The compiled ExecuTorch program object.

    Raises:
        RuntimeError: If any stage of conversion fails.
    """
    # Ensure model is in evaluation mode for consistent export
    model = torch.ao.quantization.move_exported_model_to_eval(model)

    # Step 1: Export to ATen dialect graph via Torch Export API
    try:
        aten_graph = torch.export.export(model, example_inputs)
    except Exception as e:
        logging.error("Failed to export model to ATen dialect: %s", e)
        raise RuntimeError("ATen export failed") from e

    if verbose:
        print("=== ATen Dialect Graph ===")
        print(aten_graph)

    # Step 2: Perform edge compilation transforms and lowering
    try:
        edge_program = to_edge_transform_and_lower(
            aten_graph,
            partitioner=[XnnpackPartitioner()],
        )
    except Exception as e:
        logging.error("Edge compilation failed: %s", e)
        raise RuntimeError("Edge compilation failed") from e

    if verbose:
        print("=== Edge Program Graph ===")
        print(edge_program.exported_program())

    # Step 3: Convert to ExecuTorch program using specified backend config
    try:
        executorch_program = edge_program.to_executorch(
            ExecutorchBackendConfig(
                passes=[],                   # No additional custom passes
                memory_planning_pass=MemoryPlanningPass(),  # Default memory planning
            )
        )
    except Exception as e:
        logging.error("Conversion to ExecuTorch program failed: %s", e)
        raise RuntimeError("ExecuTorch conversion failed") from e

    # Step 4: Save program buffer if a path is provided
    if save_path:
        try:
            filename = save_pte_program(executorch_program, save_path)
            size_mb = os.path.getsize(filename) / 1e6
            logging.info("ExecuTorch program saved as %s (%.2f MB)", filename, size_mb)
        except Exception as e:
            logging.error("Failed to save ExecuTorch program: %s", e)
            raise RuntimeError(f"Saving .pte file failed: {save_path}") from e

    return filename

def convert_quantized_to_edge_pte(
    quantized_model: torch.nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    save_path: Optional[str] = "quantized_model.pte"
) -> str:
    """
    Compiles a quantized PyTorch model into an ExecuTorch PTE program and saves it.

    This function:
    1. Loads necessary quantized op kernels.
    2. Exports the model to an edge-optimized IR via export_to_edge.
    3. Converts the IR to an ExecuTorch program
    4. Writes the resulting ExecuTorch program to disk if a save path is provided.

    Args:
        quantized_model (torch.nn.Module): Quantized PyTorch model to compile.
        example_inputs (Tuple[torch.Tensor, ...]): Example input(s) for tracing.
        save_path (Optional[str]): Path where the .pte file will be saved.

    Returns:
        str: Filename of the saved .pte program.

    Raises:
        RuntimeError: If any compilation or saving step fails.
    """
    # Step 1: Ensure quantized ops are registered (will raise if missing)
    try:
        _ = torch.ops.quantized_decomposed.add.out
    except AttributeError:
        logging.warning("No quantized ops registered; loading kernels now.")
        kernel_lib = os.path.abspath(
            os.path.join("..", "executorch", "cmake-out", "kernels", "quantized", "libquantized_ops_aot_lib.so")
        )
        torch.ops.load_library(kernel_lib)

    # Prepare edge compilation configuration
    edge_compile_config = EdgeCompileConfig(_check_ir_validity=False)

    # Step 2: Export to edge IR
    try:
        edge_program = export_to_edge(
            quantized_model,
            example_inputs,
            edge_compile_config=edge_compile_config
        )
    except Exception as e:
        logging.error("Export to edge IR failed: %s", e)
        raise RuntimeError("Edge IR export failed") from e

    # Step 3: Convert to ExecuTorch program
    try:
        executorch_program = edge_program.to_executorch(
            config=ExecutorchBackendConfig(extract_delegate_segments=False)
        )
    except Exception as e:
        logging.error("Edge to ExecuTorch conversion failed: %s", e)
        raise RuntimeError("ExecuTorch conversion failed") from e

    # Step 4: Save and report size
    if save_path:
        try:
            filename = save_pte_program(executorch_program, save_path)
            size_mb = os.path.getsize(filename) / 1e6
            logging.info("ExecuTorch program saved as %s (%.2f MB)", filename, size_mb)
        except Exception as e:
            logging.error("Saving PTE program failed: %s", e)
            raise RuntimeError("Saving PTE file failed") from e

    return filename
