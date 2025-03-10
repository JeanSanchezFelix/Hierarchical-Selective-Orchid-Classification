import torch
import torch.nn as nn
import torch.ao.ns._numeric_suite_fx as ns
import matplotlib as plt
from torch.ao.quantization import get_default_qconfig, QConfigMapping, QConfig, MinMaxObserver, prepare_qat, convert
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.utils.data import DataLoader

def calibrate_model(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> None:
    """
    Runs inference on the evaluation data to calibrate the quantization parameters.
    
    Parameters:
        model (nn.Module): The FX prepared model.
        data_loader (DataLoader): Evaluation DataLoader for calibration.
    """
    model.eval()
    with torch.no_grad():
        for images, _ in data_loader:
            # Forward pass to collect activation statistics
            model(images.to(device))

def get_custom_qconfig() -> QConfig:
    """
    Creates a custom QConfig with explicit quant_min and quant_max settings for observers.
    
    Returns:
        QConfig: The custom quantization configuration.
    """
    # Define a custom observer for activations with explicit quantization range.
    activation_observer = MinMaxObserver.with_args(quant_min=0, quant_max=255, dtype=torch.quint8)
    
    # Define a custom observer for weights with symmetric quantization range.
    weight_observer = MinMaxObserver.with_args(quant_min=-127, quant_max=127, dtype=torch.qint8)
    
    return QConfig(activation=activation_observer, weight=weight_observer)

def quantize_model(model: nn.Module, example_input: torch.Tensor, eval_loader: DataLoader) -> nn.Module:
    """
    Performs FX Graph Mode Post-Training Static Quantization on the given model.
    
    Parameters:
        model (nn.Module): The float (FP32) model to quantize.
        example_input (torch.Tensor): An example input for tracing.
        eval_loader (DataLoader): DataLoader used for calibration.
    
    Returns:
        nn.Module: The quantized model.
    """
    # Specify the quantization configuration (using default config for x86 CPUs)
    # qconfig = get_default_qconfig("x86")
    # qconfig_mapping = QConfigMapping().set_global(qconfig)

    # Use the custom qconfig in your FX quantization flow
    # model.fuse_model()
    custom_qconfig = get_custom_qconfig()
    qconfig_mapping = torch.ao.quantization.QConfigMapping().set_global(custom_qconfig)
    
    # Prepare the model for FX graph mode quantization
    prepared_model = prepare_fx(model, qconfig_mapping, example_input)
    print("Prepared model FX graph:")
    print(prepared_model.graph)
    
    # Run calibration to collect statistics
    calibrate_model(prepared_model, eval_loader)
    
    # Convert the calibrated model to a quantized version
    quantized_model = convert_fx(prepared_model)
    print("Quantized model FX graph:")
    print(quantized_model.graph)
    
    return quantized_model

def debug_quantized_model(original_model: nn.Module, quantized_model: nn.Module) -> list[int]:
    """
    Comparing Weights Using the Numeric Suite.
    
    Parameters:
        original_model (nn.Module): The original model to comapre to.
        quantized_model (nn.Module): The quantized model to inspect.
    """
    print("Debugging quantized model parameters:")

    # Extract weight pairs between the original (float) and quantized models.
    # Here we label the weights from the float model as 'float' and from the quantized model as 'quantized'.
    weight_comparison = ns.extract_weights('float', original_model, 'quantized', quantized_model)

    # Extend the comparison dictionary by computing the SQNR (Signal-to-Quantization-Noise Ratio)
    ns.extend_logger_results_with_comparison(
        weight_comparison, 'float', 'quantized', torch.ao.ns.fx.utils.compute_sqnr, 'sqnr'
    )

    sqnr_list = []

    # Now, print the SQNR values for each weight pair for inspection.
    print("Weight Comparison (SQNR):")
    for name, comp in weight_comparison.items():
        # Each 'comp' contains entries for both sides and the computed SQNR.
        sqnr = comp['weight']['quantized'][0].get('sqnr', 'N/A')
        sqnr_list.append(sqnr[0])
        print(f"{name}: SQNR = {sqnr}")

    return sqnr_list

def plot_sqnr(xdata, ydata, xlabel, ylabel, title):
    # Create the plot
    fig = plt.figure(figsize=(10, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Use plt.gca() to get the current axes (which uses data coordinates)
    ax = plt.gca()

    # Plot xdata vs ydata
    ax.plot(xdata, ydata)

    print(min(ydata))

    # Set the x and y axis limits to their respective min and max values
    ax.set_xlim([min(xdata)-1, max(xdata)+1])
    ax.set_ylim([min(ydata)-1, max(ydata)+1])

    # Set the x and y ticks to increment by 1
    ax.set_xticks(range(min(xdata), max(xdata), 2))  # Set x ticks from min to max with step 2
    # ax.set_yticks(range(min(ydata), max(ydata) + 1, 5))  # Set y ticks from min to max with step 1

    plt.tight_layout()

    # Show the plot
    plt.show()
