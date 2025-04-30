# model_compression/src/utils/__init__.py
"""
Utility package initialization.
Exports core functions for data handling, model setup, logging, parsing, metrics, benchmarking, and callbacks.
"""
# Model setup
from .model_setup import (
    setup_model,
    setup_criterion,
    setup_optimizer,
    tf_setup
)

# Data preprocessing and imbalance
from .preprocessing import load_data

from .data_imbalance import (
    calculate_model_weights,
    get_weighted_sampler
)

# Argument parsing and configuration
from .parsing import parse

# Logging setup
from .logging_setup import configure_logging

# Metrics and plotting
from .metrics import (
    calculate_metrics,
    plot_metric_bar,
    plot_confusion_matrix,
    plot_train_val_curve,
    plot_roc_auc_curve,
    plot_calibration_curve,
    plot_log_loss  
)

# Benchmarking tools
from .benchmarking import (
    measure_inference_performance,
    calculate_speedup,
    measure_memory_usage,
    model_size_mb,
    measure_idle_power_consumption,
    measure_power_consumption,
    measure_latency_percentiles,
    measure_throughput_per_watt,
    benchmark
)

# Callback registry
from .callbacks import process_callbacks, CALLBACK_REGISTRY

__all__ = [
    'setup_model', 'setup_criterion', 'setup_optimizer', 'tf_setup',
    'load_data',
    'calculate_model_weights', 'get_weighted_sampler',
    'parse',
    'configure_logging',
    'calculate_metrics', 'plot_metric_bar', 'plot_confusion_matrix',
    'plot_train_val_curve', 'plot_roc_auc_curve', 'plot_calibration_curve', 'plot_log_loss',
    'measure_inference_performance', 'calculate_speedup',
    'measure_memory_usage', 'model_size_mb', 'measure_idle_power_consumption',
    'measure_power_consumption', 'measure_latency_percentiles', 'measure_throughput_per_watt',
    'benchmark',
    'process_callbacks', 'CALLBACK_REGISTRY'
]
