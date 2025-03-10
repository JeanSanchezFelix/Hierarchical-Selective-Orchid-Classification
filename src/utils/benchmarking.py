import os
import time
import logging
import torch
import torch.nn as nn
from typing import Tuple, Dict
from torch.utils.data import DataLoader

try:
    import psutil
except ImportError:
    psutil = None  # psutil is optional for CPU memory measurement

def measure_inference_performance(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_warmup: int = 5,
    num_trials: int = 50
) -> Tuple[float, float]:
    """
    Measures the average inference time per sample and throughput for the given model.

    The function first performs a number of warm-up iterations to ensure the model and device 
    are ready, then records the inference time over a specified number of trials.

    Parameters:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader to supply inference data.
        device (torch.device): Device to perform inference on (CPU or GPU).
        num_warmup (int): Number of warm-up iterations (batches) before timing.
        num_trials (int): Number of batches to run for timing inference.

    Returns:
        Tuple[float, float]: A tuple containing:
            - avg_time_per_sample (float): Average inference time per sample in seconds.
            - throughput (float): Number of samples processed per second.
    """
    model.eval()
    total_time = 0.0
    total_samples = len(data_loader.dataset)

    # Warm-up loop: run a few iterations without recording time.
    with torch.no_grad():
        warmup_iter = iter(data_loader)
        for _ in range(num_warmup):
            try:
                inputs, _ = next(warmup_iter)
            except StopIteration:
                # Restart the iterator if necessary.
                warmup_iter = iter(data_loader)
                inputs, _ = next(warmup_iter)
            inputs = inputs.to(device)
            _ = model(inputs)

    # Timing loop: run for a fixed number of batches.
    trial_iter = iter(data_loader)
    with torch.no_grad():
        for _ in range(num_trials):
            try:
                inputs, _ = next(trial_iter)
            except StopIteration:
                trial_iter = iter(data_loader)
                inputs, _ = next(trial_iter)
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            start_time = time.time()
            _ = model(inputs)
            elapsed = time.time() - start_time
            total_time += elapsed

    avg_time_per_sample = total_time / total_samples
    throughput = total_samples / total_time

    logging.info(f"Average inference time per sample: {avg_time_per_sample:.6f} seconds")
    logging.info(f"Throughput: {throughput:.2f} samples/second")

    return avg_time_per_sample, throughput

def calculate_speedup(
    baseline_time: float,
    baseline_throughput: float,
    target_time: float,
    target_throughput: float
) -> Dict[str, float]:
    """
    Calculates the speedup between two models based on their average inference times and throughput.

    Speedup is calculated as:
        - time_speedup = baseline_time / target_time
        - throughput_speedup = target_throughput / baseline_throughput

    Parameters:
        baseline_time (float): Average inference time per sample for the baseline model.
        baseline_throughput (float): Throughput for the baseline model (samples per second).
        target_time (float): Average inference time per sample for the target model.
        target_throughput (float): Throughput for the target model (samples per second).

    Returns:
        Dict[str, float]: Dictionary with keys 'time_speedup' and 'throughput_speedup'.
    """
    if target_time == 0 or baseline_throughput == 0:
        raise ValueError("Target time and baseline throughput must be non-zero for speedup calculation.")

    time_speedup = baseline_time / target_time
    throughput_speedup = target_throughput / baseline_throughput

    logging.info(f"Time speedup: {time_speedup:.2f}x")
    logging.info(f"Throughput speedup: {throughput_speedup:.2f}x")

    return {
        "time_speedup": time_speedup,
        "throughput_speedup": throughput_speedup
    }

def measure_memory_usage(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_batches: int = 10
) -> float:
    """
    Measures the peak memory usage during inference.

    For CUDA devices, it resets the peak memory counter and then performs inference over a few batches.
    For CPU inference, if psutil is available, it returns the process memory usage (in MB).

    Parameters:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader to supply inference data.
        device (torch.device): Device on which inference is performed.
        num_batches (int): Number of batches to run for measuring memory usage.

    Returns:
        float: Peak memory usage in MB.
    """
    model.eval()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            batch_iter = iter(data_loader)
            for _ in range(num_batches):
                try:
                    inputs, _ = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(data_loader)
                    inputs, _ = next(batch_iter)
                inputs = inputs.to(device)
                _ = model(inputs)
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        logging.info(f"Peak GPU memory usage: {peak_memory:.2f} MB")
    else:
        # For CPU, if psutil is available, return the current process memory usage.
        if psutil is not None:
            process = psutil.Process(os.getpid())
            mem_bytes = process.memory_info().rss  # Resident Set Size in bytes
            peak_memory = mem_bytes / (1024 ** 2)
            logging.info(f"Process memory usage (CPU): {peak_memory:.2f} MB")
        else:
            peak_memory = 0.0
            logging.warning("psutil not available; cannot measure CPU memory usage.")
    return peak_memory

def model_size(model: torch.nn.Module) -> None:
    """
    Return the size of the given model.
    
    Parameters:
        model (nn.Module): The model to inspect.

    Returns:
        size (float): The size of the model in MB.
    """
    try:
        # Convert model to TorchScript if it isn't already
        scripted_model = model if isinstance(model, torch.jit.ScriptModule) else torch.jit.script(model)
        # Save the TorchScript model
        torch.jit.save(scripted_model, "temp_model.pt")
        
        # Save the full model (including architecture) in .pth format
        torch.save(model, "temp.pth")

        script_size = os.path.getsize('temp_model.pt') / 1e6
        pth_size = os.path.getsize('temp.pth') / 1e6
        
        # Get and print model sizes in MB
        print(f"Model size as TorchScript: {script_size:.4f} MB")
        print(f"Model size as Pth: {pth_size:.4f} MB")
    
    finally:
        # Clean up temporary files
        if os.path.exists("temp_model.pt"):
            os.remove("temp_model.pt")
        if os.path.exists("temp.pth"):
            os.remove("temp.pth")

    return script_size, pth_size


# Example usage:
# Assume model, test_loader, device are defined.
# avg_time1, throughput1 = measure_inference_performance(model1, test_loader, device)
# avg_time2, throughput2 = measure_inference_performance(model2, test_loader, device)
# speedup = calculate_speedup(avg_time1, throughput1, avg_time2, throughput2)
# memory_usage = measure_memory_usage(model1, test_loader, device)
