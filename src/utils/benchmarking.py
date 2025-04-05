import os
import time
import random
import subprocess
import threading
import logging
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from src.Quantization.quantization_utils.conversions.litert import is_quantized_model

try:
    import psutil
except ImportError:
    psutil = None  # psutil is optional for CPU memory measurement

try:
    import pyRAPL
except ImportError:
    pyRAPL = None

def seed_everything(seed: int = 42):
    """
    Seeds random number generators for reproducibility.

    Parameters:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def warm_up(model: nn.Module, dataloader: DataLoader, device: torch.device, num_warmup: int = 5) -> None:
    """
    Runs a warm-up phase to stabilize the model and device before benchmarking.

    Parameters:
        model (nn.Module): The model to warm up.
        dataloader (DataLoader): DataLoader supplying input data.
        device (torch.device): Device on which to run inference.
        num_warmup (int): Number of warm-up batches.
    """
    # Set model to evaluation mode.
    if is_quantized_model(model):
        torch.ao.quantization.move_exported_model_to_eval(model)
    else:
        model.eval()

    with torch.no_grad():
        warmup_iter = iter(dataloader)
        for _ in range(num_warmup):
            try:
                inputs, _ = next(warmup_iter)
            except StopIteration:
                warmup_iter = iter(dataloader)
                inputs, _ = next(warmup_iter)
            inputs = inputs.to(device)
            _ = model(inputs)

def measure_inference_performance(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_warmup: int = 5,
    num_trials: int = 50
) -> tuple[float, float]:
    """
    Measures average inference time per sample and throughput.

    Parameters:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader supplying inference data.
        device (torch.device): Device on which to run inference.
        num_warmup (int): Warm-up iterations before timing.
        num_trials (int): Number of batches for timing inference.

    Returns:
        tuple[float, float]: (avg_time_per_sample in seconds, throughput in samples/sec)
    """
    # Set model to evaluation mode.
    if is_quantized_model(model):
        torch.ao.quantization.move_exported_model_to_eval(model)
    else:
        model.eval()
        
    total_time = 0.0
    total_samples = len(dataloader.dataset)

    # Warm up before timing.
    warm_up(model, dataloader, device, num_warmup=num_warmup)

    # Set model to evaluation mode.
    if is_quantized_model(model):
        torch.ao.quantization.move_exported_model_to_eval(model)
    else:
        model.eval()

    total_time = 0.0
    total_samples = len(dataloader.dataset)

    # Run a fixed number of batches
    trial_iter = iter(dataloader)
    with torch.no_grad():
        for _ in range(num_trials):
            try:
                inputs, _ = next(trial_iter)
            except StopIteration:
                trial_iter = iter(dataloader)
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
) -> dict[str, float]:
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
        dict[str, float]: Dictionary with keys 'time_speedup' and 'throughput_speedup'.
    """
    if target_time == 0 or baseline_throughput == 0:
        raise ValueError("Target time and baseline throughput must be non-zero for speedup calculation.")

    time_speedup = baseline_time / target_time
    throughput_speedup = target_throughput / baseline_throughput

    logging.info(f"Time speedup: {time_speedup:.2f}x")
    logging.info(f"Throughput speedup: {throughput_speedup:.2f}x")
    return {"time_speedup": time_speedup, "throughput_speedup": throughput_speedup}


def measure_memory_usage(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_warmup: int = 5,
    num_batches: int = 10
) -> float:
    """
    Measures the peak memory usage during inference.

    For CUDA devices, it resets the peak memory counter and then performs inference over a few batches.
    For CPU inference, if psutil is available, it returns the process memory usage (in MB).

    Parameters:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader to supply inference data.
        device (torch.device): Device on which inference is performed.
        num_warmup (int): Warm-up iterations before measurement.
        num_batches (int): Number of batches to run for measuring memory usage.

    Returns:
        float: Peak memory usage in MB.
    """
    warm_up(model, dataloader, device, num_warmup=num_warmup)
    # Set model to evaluation mode.
    if is_quantized_model(model):
        torch.ao.quantization.move_exported_model_to_eval(model)
    else:
        model.eval()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            batch_iter = iter(dataloader)
            for _ in range(num_batches):
                try:
                    inputs, _ = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(dataloader)
                    inputs, _ = next(batch_iter)
                inputs = inputs.to(device)
                _ = model(inputs)
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        logging.info(f"Peak GPU memory usage: {peak_memory:.2f} MB")
    else:
        if psutil is not None:
            process = psutil.Process(os.getpid())
            mem_bytes = process.memory_info().rss
            peak_memory = mem_bytes / (1024 ** 2)
            logging.info(f"Process memory usage (CPU): {peak_memory:.2f} MB")
        else:
            peak_memory = 0.0
            logging.warning("psutil not available; cannot measure CPU memory usage.")
    return peak_memory

def model_size(model: nn.Module) -> float:
    """
    Computes the model size (state_dict size) in MB.

    Parameters:
        model (torch.nn.Module): The model to be evaluated.

    Returns:
        float: Model size in MB.
    """
    temp_pth_path = "temp.pth"
    try:
        torch.save(model.state_dict(), temp_pth_path)
        pth_size = os.path.getsize(temp_pth_path) / 1e6 if os.path.exists(temp_pth_path) else 0.0
        return pth_size
    finally:
        if os.path.exists(temp_pth_path):
            os.remove(temp_pth_path)

def _gpu_power_sampler(stop_event: threading.Event, sampling_interval: float, results: list):
    """
    Samples GPU power consumption using nvidia-smi at regular intervals and appends values to results.
    """
    while not stop_event.is_set():
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            )
            power = float(output.splitlines()[0].strip())
        except Exception as e:
            logging.warning(f"Error sampling GPU power: {e}")
            power = 0.0
        results.append(power)
        time.sleep(sampling_interval)

def measure_idle_power_consumption(
    device: torch.device,
    idle_duration: float = 5.0,
    sampling_interval: float = 0.1
) -> float:
    """
    Measures the average idle power consumption (in Watts) over a specified duration.

    For CUDA devices, it samples power using nvidia-smi in a background thread.
    For CPU devices, if pyRAPL is available, it uses pyRAPL to measure energy consumption while idle.

    Args:
        device (torch.device): Device on which to measure power consumption.
        idle_duration (float): Duration (in seconds) over which to measure idle power.
        sampling_interval (float): Time interval (in seconds) between power samples (for GPU).

    Returns:
        float: Average idle power consumption in Watts.
    """
    # For GPU, use nvidia-smi to sample power during idle.
    if device.type == "cuda":
        power_samples: list[float] = []
        stop_event = threading.Event()
        # Start the background thread for power sampling.
        sampler_thread = threading.Thread(target=_gpu_power_sampler, args=(stop_event, sampling_interval, power_samples))
        sampler_thread.start()
        # Sleep for the idle duration to let the GPU settle in idle.
        time.sleep(idle_duration)
        # Signal the sampling thread to stop and wait for it.
        stop_event.set()
        sampler_thread.join()
        avg_idle_power = sum(power_samples) / len(power_samples) if power_samples else 0.0
        logging.info(f"Average GPU idle power consumption: {avg_idle_power:.2f} Watts")
        return avg_idle_power

    # For CPU, attempt to use pyRAPL if available.
    else:
        if pyRAPL is None:
            logging.warning("pyRAPL is not available; cannot measure CPU power consumption.")
            return 0.0
        try:
            pyRAPL.setup()
            meter = pyRAPL.Measurement('inference')
            start_time = time.time()
            # Measure energy consumption while idle.
            with meter:
                time.sleep(idle_duration)
            total_time = time.time() - start_time
            # Convert energy from microjoules to joules.
            total_energy = (meter.result.pkg if hasattr(meter.result, 'pkg') else 0.0)
            avg_idle_power = (total_energy[0] / total_time if total_time > 0 else 0.0) / 1e6
            logging.info(f"Average CPU idle power consumption (pyRAPL): {avg_idle_power:.2f} Watts")
            return avg_idle_power
        except PermissionError as e:
            logging.warning(f"pyRAPL permission error: {e}")
        except Exception as e:
            logging.warning(f"pyRAPL measurement failed: {e}")


def measure_power_consumption(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_warmup: int = 5,
    num_batches: int = 10,
    sampling_interval: float = 0.1
) -> float:
    """
    Measures average power consumption during inference in Watts.

    For CUDA devices, uses nvidia-smi with a background thread.
    For CPU devices, if pyRAPL is available, uses it to measure energy consumption.
    A warm-up phase is performed before measurement.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader supplying inference data.
        device (torch.device): Device on which inference is performed.
        num_warmup (int): Number of warm-up batches.
        num_batches (int): Number of batches for measurement.
        sampling_interval (float): Interval between power samples (for GPU).

    Returns:
        float: Average power consumption in Watts.
    """
    # Warm up the model
    warm_up(model, dataloader, device, num_warmup=num_warmup)
    
    # Set model to evaluation mode.
    if is_quantized_model(model):
        torch.ao.quantization.move_exported_model_to_eval(model)
    else:
        model.eval()

    # GPU measurement.
    if device.type == "cuda":
        power_samples = []
        stop_event = threading.Event()
        sampler_thread = threading.Thread(target=_gpu_power_sampler, args=(stop_event, sampling_interval, power_samples))
        sampler_thread.start()

        with torch.no_grad():
            trial_iter = iter(dataloader)
            for _ in range(num_batches):
                try:
                    inputs, _ = next(trial_iter)
                except StopIteration:
                    trial_iter = iter(dataloader)
                    inputs, _ = next(trial_iter)
                inputs = inputs.to(device)
                _ = model(inputs)
        stop_event.set()
        sampler_thread.join()
        avg_power = sum(power_samples) / len(power_samples) if power_samples else 0.0
        logging.info(f"Average GPU power consumption: {avg_power:.2f} Watts")
        return avg_power

    # CPU measurement using pyRAPL.
    else:
        if pyRAPL is None:
            logging.warning("pyRAPL is not available; cannot measure CPU power consumption.")
            return 0.0
        try:
            pyRAPL.setup()
            meter = pyRAPL.Measurement('inference')
            start_time = time.time()
            with meter:
                trial_iter = iter(dataloader)
                for _ in range(num_batches):
                    try:
                        inputs, _ = next(trial_iter)
                    except StopIteration:
                        trial_iter = iter(dataloader)
                        inputs, _ = next(trial_iter)
                    _ = model(inputs)
            total_time = time.time() - start_time
            # Convert energy from microjoules to joules.
            total_energy = (meter.result.pkg if hasattr(meter.result, 'pkg') else 0.0)
            avg_power = (total_energy[0] / total_time if total_time > 0 else 0.0) / 1e6
            logging.info(f"Average CPU power consumption: {avg_power:.2f} Watts")
            return avg_power
        except PermissionError as e:
            logging.warning(f"pyRAPL permission error: {e}")
        except Exception as e:
            logging.warning(f"pyRAPL measurement failed: {e}")


def measure_latency_percentiles(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_trials: int = 50
) -> dict[str, float]:
    """
    Measures the latency percentiles (P50, P95, P99) of inference time per batch.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader supplying inference data.
        device (torch.device): Device on which to run inference.
        num_trials (int): Number of inference trials to record. Defaults to 50.

    Returns:
        dict[str, float]: Dictionary with keys 'p50', 'p95', and 'p99' representing latency percentiles in seconds.
    """
    warm_up(model, dataloader, device)

    # Set model to evaluation mode.
    if is_quantized_model(model):
        torch.ao.quantization.move_exported_model_to_eval(model)
    else:
        model.eval()
        
    latencies: list[float] = []
    
    with torch.no_grad():
        # Record inference time for a fixed number of trials.
        for _ in range(num_trials):
            inputs, _ = next(iter(dataloader))
            inputs = inputs.to(device)
            start_time = time.time()
            _ = model(inputs)
            elapsed = time.time() - start_time
            latencies.append(elapsed)
    
    # Compute latency percentiles.
    percentiles = {
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
    }
    
    logging.info(f"Latency percentiles: P50={percentiles['p50']:.6f}s, P95={percentiles['p95']:.6f}s, P99={percentiles['p99']:.6f}s")
    return percentiles


def measure_throughput_per_watt(throughput: float, power: float) -> float:
    """
    Computes throughput per Watt, indicating energy efficiency.

    Args:
        throughput (float): Throughput in samples per second.
        power (float): Average power consumption in Watts.

    Returns:
        float: Throughput per Watt (samples/sec/Watt).
    """
    if power > 0:
        throughput_per_watt = throughput / power
        logging.info(f"Throughput per Watt: {throughput_per_watt:.2f} samples/sec/Watt")
        return throughput_per_watt
    logging.warning("Power consumption is zero, cannot compute Throughput per Watt.")
    return 0.0


def benchmark(model1: nn.Module, model2: nn.Module, dataloader: DataLoader, device: torch.device) -> None:
    """
    Benchmarks two models across various metrics including size, inference performance, memory usage,
    power consumption, energy per sample, latency percentiles, and throughput per Watt.
    Results are printed in a formatted table using pandas.

    Args:
        model1 (nn.Module): The first model (e.g., teacher model).
        model2 (nn.Module): The second model (e.g., student model).
        dataloader (DataLoader): DataLoader supplying input data.
        device (torch.device): Device on which to run inference.
    """
    # Measure model sizes.
    model1_size = model_size(model1)
    model2_size = model_size(model2)

    # Measure inference performance.
    avg_time1, throughput1 = measure_inference_performance(model1, dataloader, device)
    avg_time2, throughput2 = measure_inference_performance(model2, dataloader, device)
    speedup = calculate_speedup(avg_time1, throughput1, avg_time2, throughput2)

    # Measure memory usage.
    mem_usage1 = measure_memory_usage(model1, dataloader, device)
    mem_usage2 = measure_memory_usage(model2, dataloader, device)

    # Idle power consumption
    idle_power = measure_idle_power_consumption(device=device, idle_duration=5.0, sampling_interval=0.1)

    # Measure power consumption.
    power1 = measure_power_consumption(model1, dataloader, device)
    power2 = measure_power_consumption(model2, dataloader, device)

    # Compute energy efficiency: Joules per sample = (avg power in Watts * avg inference time in sec)
    energy_efficiency1 = power1 * avg_time1
    energy_efficiency2 = power2 * avg_time2

    # Measure latency percentiles.
    latency1 = measure_latency_percentiles(model1, dataloader, device)
    latency2 = measure_latency_percentiles(model2, dataloader, device)

    # Compute throughput per Watt.
    tpw1 = measure_throughput_per_watt(throughput1, power1)
    tpw2 = measure_throughput_per_watt(throughput2, power2)

    # Create a DataFrame to display results.
    data = {
        "Metric": [
            "Model Size (MB)",
            "Inference Time (sec/sample)",
            "Throughput (samples/sec)",
            "Memory Usage (MB)",
            "Idle Power (Watts)",
            "Avg Power (Watts)",
            "Energy per Sample (Joules)",
            "Throughput per Watt (samples/sec/W)",
            "Latency P50 (sec)",
            "Latency P95 (sec)",
            "Latency P99 (sec)",
            "Time Speedup",
            "Throughput Speedup"
        ],
        "Model 1": [
            f"{model1_size:.4f}",
            f"{avg_time1:.6f}",
            f"{throughput1:.2f}",
            f"{mem_usage1:.2f}",
            f"{idle_power:.2f}",
            f"{power1:.2f}",
            f"{energy_efficiency1:.6f}",
            f"{tpw1:.2f}",
            f"{latency1['p50']:.6f}",
            f"{latency1['p95']:.6f}",
            f"{latency1['p99']:.6f}",
            f"{speedup.get('time_speedup', 'N/A'):.2f}x",
            f"{speedup.get('throughput_speedup', 'N/A'):.2f}x"
        ],
        "Model 2": [
            f"{model2_size:.4f}",
            f"{avg_time2:.6f}",
            f"{throughput2:.2f}",
            f"{mem_usage2:.2f}",
            "-",
            f"{power2:.2f}",
            f"{energy_efficiency2:.6f}",
            f"{tpw2:.2f}",
            f"{latency2['p50']:.6f}",
            f"{latency2['p95']:.6f}",
            f"{latency2['p99']:.6f}",
            "-",
            "-"
        ]
    }
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
