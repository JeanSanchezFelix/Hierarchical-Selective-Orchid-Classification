import os
import time
import random
import subprocess
import threading
import logging
from typing import Tuple, Optional, List, Dict

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from model_compression.src.quantization.utils.inspect import is_quantized_model

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pyRAPL
except ImportError:
    pyRAPL = None

def _seed_everything(seed: int = 42) -> None:
    """
    Seed all relevant random number generators for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _warm_up(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_warmup: int = 5
) -> None:
    """
    Run a few inference batches without timing to stabilize execution.

    Args:
        model: PyTorch model.
        dataloader: DataLoader for inference data.
        device: Torch device.
        num_warmup: Number of batches to run.
    """
    # Set model to evaluation mode.
    model_to_eval = (
        torch.ao.quantization.move_exported_model_to_eval(model)
        if is_quantized_model(model) else model
    )
    model_to_eval.to(device).eval()

    it = iter(dataloader)
    with torch.no_grad():
        for _ in range(num_warmup):
            try:
                inputs, _ = next(it)
            except StopIteration:
                it = iter(dataloader)
                inputs, _ = next(it)
            _ = model_to_eval(inputs.to(device))

def measure_inference_performance(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_warmup: int = 5,
    num_trials: int = 50
) -> Tuple[float, float]:
    """
    Measure average inference time per sample and overall throughput.

    Args:
        model: PyTorch model.
        dataloader: DataLoader for inference data.
        device: Torch device.
        num_warmup: Warm-up iterations.
        num_trials: Number of batches to time.

    Returns:
        avg_time_per_sample (sec), throughput (samples/sec).
    """
    model_to_eval = (
        torch.ao.quantization.move_exported_model_to_eval(model)
        if is_quantized_model(model) else model
    )
    model_to_eval.to(device).eval()

    total_samples = len(dataloader.dataset)
    _warm_up(model_to_eval, dataloader, device, num_warmup)

    total_time = 0.0
    it = iter(dataloader)
    with torch.no_grad():
        for _ in range(num_trials):
            try:
                inputs, _ = next(it)
            except StopIteration:
                it = iter(dataloader)
                inputs, _ = next(it)
            batch = inputs.to(device)
            start = time.time()
            _ = model_to_eval(batch)
            total_time += time.time() - start

    avg_time = total_time / total_samples
    throughput = total_samples / total_time if total_time > 0 else float('inf')
    logging.info(f"Avg inference time/sample: {avg_time:.6f}s, Throughput: {throughput:.2f} samples/s")
    return avg_time, throughput

def calculate_speedup(
    baseline_time: float,
    baseline_throughput: float,
    target_time: float,
    target_throughput: float
) -> dict:
    """
    Compute speedup metrics comparing a target model against a baseline.

    Args:
        baseline_time: Avg time/sample for baseline.
        baseline_throughput: Throughput for baseline.
        target_time: Avg time/sample for target.
        target_throughput: Throughput for target.

    Returns:
        Dict with 'time_speedup', 'throughput_speedup'.

    Raises:
        ValueError: If zero or negative values encountered.
    """
    if baseline_time <= 0 or target_time <= 0:
        raise ValueError("Times must be > 0.")
    if baseline_throughput <= 0 or target_throughput <= 0:
        raise ValueError("Throughput must be > 0.")

    time_speedup = baseline_time / target_time
    throughput_speedup = target_throughput / baseline_throughput
    logging.info(f"Time speedup: {time_speedup:.2f}x, Throughput speedup: {throughput_speedup:.2f}x")
    return {"time_speedup": time_speedup, "throughput_speedup": throughput_speedup}


def measure_memory_usage(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_warmup: int = 5,
    num_batches: int = 10
) -> float:
    """
    Measure peak memory usage (GPU or CPU) during inference.

    Args:
        model: PyTorch model.
        dataloader: DataLoader for inference data.
        device: Torch device.
        num_warmup: Warm-up iterations.
        num_batches: Batches to measure.

    Returns:
        Peak memory in MB.
    """
    _warm_up(model, dataloader, device, num_warmup)
    model_to_eval = (
        torch.ao.quantization.move_exported_model_to_eval(model)
        if is_quantized_model(model) else model
    )
    model_to_eval.to(device).eval()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            it = iter(dataloader)
            for _ in range(num_batches):
                try:
                    inputs, _ = next(it)
                except StopIteration:
                    it = iter(dataloader)
                    inputs, _ = next(it)
                _ = model_to_eval(inputs.to(device))
        peak = torch.cuda.max_memory_allocated(device) / 1e6
        logging.info(f"Peak GPU memory: {peak:.2f} MB")
    else:
        if psutil:
            proc = psutil.Process(os.getpid())
            peak = proc.memory_info().rss / 1e6
            logging.info(f"CPU memory usage: {peak:.2f} MB")
        else:
            peak = 0.0
            logging.warning("psutil not installed; CPU memory not measured.")
    return peak

def model_size_mb(model: nn.Module, temp_path: str = "temp.pth") -> float:
    """
    Compute disk size of model state dict in MB.

    Args:
        model: PyTorch model.
        temp_path: Temporary file path for saving.

    Returns:
        Size in MB.
    """
    try:
        torch.save(model.state_dict(), temp_path)
        size_mb = os.path.getsize(temp_path) / 1e6
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    logging.info(f"Model size: {size_mb:.2f} MB")
    return size_mb

def _gpu_power_sampler(
    stop_event: threading.Event,
    sampling_interval: float,
    results: List[float]
) -> None:
    """
    Sample GPU power draw via nvidia-smi until stopped.

    Args:
        stop_event: Event to signal stop.
        sampling_interval: Seconds between samples.
        results: List to append power readings.
    """
    while not stop_event.is_set():
        try:
            output = subprocess.check_output([
                "nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"
            ], encoding='utf-8')
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
        device: Torch device.
        idle_duration: Seconds to measure.
        sampling_interval: Interval between GPU samples.

    Returns:
        Average idle power consumption in Watts.
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

    Args:
        model: Model to evaluate.
        dataloader: DataLoader.
        device: Torch device.
        num_warmup: Warm-up batches.
        num_batches: Batches to measure.
        sampling_interval: GPU sampling interval.

    Returns:
        Avg power in Watts.
    """
    # Warm up the model
    _warm_up(model, dataloader, device, num_warmup=num_warmup)
    
    # Set model to evaluation mode.
    model_to_eval = (
        torch.ao.quantization.move_exported_model_to_eval(model)
        if is_quantized_model(model) else model
    )
    model_to_eval.to(device).eval()

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
        model: Model to evaluate.
        dataloader: DataLoader.
        device: Torch device.
        num_trials: Number of trials.

    Returns:
        Dict with 'p50', 'p95', 'p99'.
    """
    _warm_up(model, dataloader, device)

    # Set model to evaluation mode.
    model_to_eval = (
        torch.ao.quantization.move_exported_model_to_eval(model)
        if is_quantized_model(model) else model
    )
    model_to_eval.to(device).eval()
        
    latencies: list[float] = []
    
    with torch.no_grad():
        # Record inference time for a fixed number of trials.
        for _ in range(num_trials):
            try:
                inputs, _ = next(it)
            except StopIteration:
                it = iter(dataloader)
                inputs, _ = next(it)
            start = time.time()
            _ = model_to_eval(inputs.to(device))
            latencies.append(time.time() - start)
    
    # Compute latency percentiles.
    percentiles = {
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
    }
    
    logging.info(f"Latency percentiles: P50={percentiles['p50']:.6f}s, P95={percentiles['p95']:.6f}s, P99={percentiles['p99']:.6f}s")
    return percentiles


def measure_throughput_per_watt(
    throughput: float,
    power: float
) -> float:
    """
    Compute throughput per Watt to assess energy efficiency.

    Args:
        throughput: Samples per second.
        power: Watts consumed.

    Returns:
        Samples/sec/Watt.
    """
    if power <= 0:
        logging.warning("Power <= 0, cannot compute throughput per Watt.")
        return 0.0
    tpw = throughput / power
    logging.info(f"Throughput per Watt: {tpw:.2f} samples/s/W")
    return tpw


def benchmark(    
    model1: nn.Module,
    model2: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> None:
    """
    Benchmarks two models across various metrics including size, inference performance, memory usage,
    power consumption, energy per sample, latency percentiles, and throughput per Watt.
    Results are printed in a formatted table using pandas.

    Args:
        model1: First model.
        model2: Second model.
        dataloader: DataLoader for inference.
        device: Execution device.
    """
    import pandas as pd

    # Measure model sizes.
    model1_size = model_size_mb(model1)
    model2_size = model_size_mb(model2)

    # Measure inference performance.
    avg_time1, throughput1 = measure_inference_performance(model1, dataloader, device)
    avg_time2, throughput2 = measure_inference_performance(model2, dataloader, device)
    speedup = calculate_speedup(avg_time1, throughput1, avg_time2, throughput2)

    # Measure memory usage.
    mem_usage1 = measure_memory_usage(model1, dataloader, device)
    mem_usage2 = measure_memory_usage(model2, dataloader, device)

    # Idle power consumption
    idle_power = measure_idle_power_consumption(device=device)

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
