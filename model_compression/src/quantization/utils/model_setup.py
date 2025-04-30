import torch
import torch.nn as nn
import torchvision.models.quantization as quant_models
from typing import Optional

from torch.utils.data import DataLoader
from torch.ao.quantization import QConfig, MinMaxObserver, MovingAverageMinMaxObserver, HistogramObserver
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config

from model_compression.src.utils.model_setup import setup_model, setup_criterion, setup_optimizer

def setup_qat_student_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Loads a quantization-aware student model for knowledge distillation.
    
    This function always loads the quantization-compatible version from
    torchvision.models.quantization with default weights. For binary classification 
    (num_classes == 2), the final classification layer is modified to output a single logit;
    otherwise, it is set to match the number of classes.
    
    Supported models for quantization:
        - googlenet
        - inception_v3
        - mobilenet_v2
        - mobilenet_v3_large
        - resnet18
        - resnet50
        - resnext101_32x8d
        - shufflenet_v2_x0_5
        - shufflenet_v2_x1_0
        - shufflenet_v2_x1_5
        - shufflenet_v2_x2_0
        
    Parameters:
        model_name (str): Name of the model.
        num_classes (int): Number of output classes.
        
    Returns:
        nn.Module: The configured quantization-aware student model.
        
    Raises:
        ValueError: If an unsupported model name is given.
    """
    weights_arg = 'DEFAULT'
    model_name_lower = model_name.lower()
    
    # Supported models for quantization.
    supported_models = {
        "googlenet", "inception_v3", "mobilenet_v2", "mobilenet_v3_large",
        "resnet18", "resnet50", "resnext101_32x8d",
        "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0"
    }
    
    if model_name_lower not in supported_models:
        raise ValueError(f"Unsupported model for quantization: {model_name}")
    
    # Always load the student (quantization-aware) model.
    if model_name_lower == "googlenet":
        model = quant_models.googlenet(weights=weights_arg)
    elif model_name_lower == "inception_v3":
        model = quant_models.inception_v3(weights=weights_arg)
    elif model_name_lower == "mobilenet_v2":
        model = quant_models.mobilenet_v2(weights=weights_arg)
    elif model_name_lower == "mobilenet_v3_large":
        model = quant_models.mobilenet_v3_large(weights=weights_arg)
    elif model_name_lower == "resnet18":
        model = quant_models.resnet18(weights=weights_arg)
    elif model_name_lower == "resnet50":
        model = quant_models.resnet50(weights=weights_arg)
    elif model_name_lower == "resnext101_32x8d":
        model = quant_models.resnext101_32x8d(weights=weights_arg)
    elif model_name_lower.startswith("shufflenet_v2"):
        model = getattr(quant_models, model_name_lower)(weights=weights_arg)
    
    # Update the classification head based on model architecture.
    if model_name_lower in {"googlenet", "inception_v3", "resnet18", "resnet50", "resnext101_32x8d",
                            "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0"}:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1) if num_classes == 2 else nn.Linear(in_features, num_classes)
    elif model_name_lower == "mobilenet_v2":
        in_features = model.last_channel
        model.classifier[1] = nn.Linear(in_features, 1) if num_classes == 2 else nn.Linear(in_features, num_classes)
    elif model_name_lower == "mobilenet_v3_large":
        in_features = model.classifier[0].in_features
        model.classifier[0] = nn.Linear(in_features, 1) if num_classes == 2 else nn.Linear(in_features, num_classes)
    
    return model

#TODO: Fix custom configs to deal with UserWarning. They do not match the source code.
def get_custom_qconfig(config: str) -> QConfig:
    """
    Creates a custom quantization configuration (QConfig) for QAT based on the specified configuration.

    The configuration can be tailored for specific deployment scenarios:
    
      - "x86": For server/desktop CPUs. Uses per-tensor observers (compatible with FX conversion).
      - "qnnpack": For mobile (ARM) devices. Uses moving average observers for smoother calibration.
      - "fbgemm": For server inference. (Default fbgemm often uses per-channel quantization for weights,
                  but here we force per-tensor for FX compatibility; adjust if needed.)
      - "edge": For edge devices. Configured for a balance between model size and inference efficiency.

    Parameters:
        config (str): The desired quantization configuration identifier.
    
    Returns:
        QConfig: The corresponding quantization configuration.
    """
    if config == "x86":
        # Use per-tensor observers which are supported by FX conversion.
        activation_observer = MinMaxObserver.with_args(
            quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine
        )
        weight_observer = MinMaxObserver.with_args(
            quant_min=-127, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        )
    elif config == "qnnpack":
        # MovingAverageMinMaxObserver is often recommended for mobile environments.
        activation_observer = MovingAverageMinMaxObserver.with_args(
            quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine
        )
        weight_observer = MovingAverageMinMaxObserver.with_args(
            quant_min=-127, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        )
    elif config == "fbgemm":
        # fbgemm is typically used in server environments. While fbgemm default config may use
        # per-channel quantization for weights, we force per-tensor for FX conversion compatibility.
        activation_observer = HistogramObserver.with_args(
            quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine
        )
        # For FX conversion compatibility, we use per_tensor_symmetric here.
        weight_observer = MinMaxObserver.with_args(
            quant_min=-127, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        )
    elif config == "edge":
        # For edge devices, one may choose observers that are tuned for low-latency and reduced model size.
        # Here we use a slightly modified configuration that might, for instance, lower the dynamic range.
        activation_observer = MinMaxObserver.with_args(
            quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine
        )
        weight_observer = MinMaxObserver.with_args(
            quant_min=-127, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        )
    else:
        # Fallback to a default configuration.
        activation_observer = MinMaxObserver.with_args(
            quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine
        )
        weight_observer = MinMaxObserver.with_args(
            quant_min=-127, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        )
    return QConfig(activation=activation_observer, weight=weight_observer)



def quantization_mode(
    model: torch.nn.Module, 
    mode: str, 
    example_inputs: Optional[tuple[torch.Tensor, ...]] = None, 
    config: str = None
) -> torch.nn.Module:
    """
    Prepares the given model for quantization-aware training (QAT) using one of three modes:
    'eager', 'fx', or 'export'.

    For 'eager' mode:
        - The model is fused where possible.
        - The default QAT QConfig for the given configuration (e.g., 'qnnpack') is assigned.
        - The model is prepared in-place using torch.ao.quantization.prepare_qat.
    For 'fx' mode:
        - A sample input is required to trace the model graph.
        - A QConfigMapping is created with the given QConfig.
        - The model is prepared using FX-based QAT preparation.
    For 'export' mode:
        - The model is exported for training and then prepared using a PT2E quantizer.
    
    Parameters:
        model (torch.nn.Module): The model to be quantized.
        mode (str): The quantization mode; valid options are 'eager', 'fx', or 'export'.
        example_inputs (Optional[tuple[torch.Tensor, ...]]): A tuple of sample input tensors required for FX Graph and Export Mode. Defaults to None.
        config (str): The configuration identifier. For example, "qnnpack" uses the default QAT QConfig 
                      for qnnpack; other values may trigger a custom QConfig.
    
    Returns:
        torch.nn.Module: The model prepared for quantization-aware training.
    
    Raises:
        ValueError: If 'fx' mode is selected without providing example_inputs, or if an invalid mode is given.
    """
    if config:
        qconfig = torch.ao.quantization.get_default_qat_qconfig(config)

    if mode == "eager":
        # Fuse the model layers where possible (for improved efficiency).
        model.fuse_model(is_qat=True)
        model.qconfig = qconfig
        # Prepare the model in-place for quantization-aware training.
        torch.ao.quantization.prepare_qat(model, inplace=True)
        print("Model prepared using Eager Mode QAT.")
    elif mode == "fx":
        # FX Graph Mode requires example inputs for tracing the model.
        if example_inputs is None:
            raise ValueError("example_inputs is required for FX Graph Mode QAT.")
        # Create a QConfigMapping with the global qconfig.
        qconfig_mapping = torch.ao.quantization.QConfigMapping().set_global(qconfig)
        model.qconfig = qconfig_mapping
        # Prepare the model using FX-based QAT preparation.
        model = prepare_qat_fx(model, qconfig_mapping, example_inputs)
        print("Model prepared using FX Graph Mode QAT.")
    elif mode == "export":
        # Export Mode requires example inputs.
        if example_inputs is None:
            raise ValueError("example_inputs is required for Export Mode QAT.")
        # Export the model for training and apply PT2E quantization.
        model = torch.export.export_for_training(model, example_inputs).module()
        # Configure the quantizer with a symmetric quantization configuration.
        operator_config = get_symmetric_quantization_config(is_per_channel=False, is_qat=True)
        quantizer = XNNPACKQuantizer().set_global(operator_config)
        model = prepare_qat_pt2e(model, quantizer)
        print("Model prepared using Export Mode QAT.")
    else:
        raise ValueError("Invalid mode. Choose either 'eager', 'fx' or 'export'.")
    
    return model


def _kd_setup(
    student: str,
    teacher: str, 
    learning_rate: float, 
    criterion_name: str, 
    optimizer_name: str, 
    dataloader: DataLoader,
    class_weights: bool = False,
    teacher_model_weights: Optional[str] = None, 
    quant_mode: Optional[str] = None,
    config: Optional[str] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> tuple[nn.Module, nn.Module, nn.Module, torch.optim.Optimizer]:
    """
    Sets up the teacher and student models, loss function, and optimizer for a combined 
    Knowledge Distillation (KD) and Quantization-Aware Training (QAT) pipeline if quant_mode os provided.

    The function performs the following steps:
      1. Loads the student model as a standard model and then prepares it for QAT using the 
         specified quantization mode (e.g., 'eager', 'fx', or 'export'). Example inputs are 
         extracted from the provided dataloader for QAT preparation.
      2. Loads the teacher model with fine-tuned weights using a separate setup function.
      3. Configures the loss function (with optional class weighting) and optimizer for training 
         the student model.

    Parameters:
        student (str): Name of the student model (to be loaded as a quantization-aware model).
        teacher (str): Name of the teacher model (to be loaded with fine-tuned weights).
        learning_rate (float): Learning rate for the optimizer.
        criterion (str): Name of the loss function to use (e.g., "cross_entropy").
        optimizer (str): Name of the optimizer to use (e.g., "adam").
        teacher_model_weights (Optional[str]): Path to the teacher model's pre-trained weights. If None,
                                               default weights are used.
        dataloader (DataLoader): DataLoader used for obtaining example inputs (and for calculating 
                                   class weights if needed).
        quant_mode (Optional[str]): Quantization mode to use for the student model ('eager', 'fx', or 'export').
                                    Defaults to None.
        config (Optional[str]): QAT configuration identifier. "qnnpack" uses the default QAT QConfig for qnnpack;
                                otherwise, a custom QConfig is used. Defaults to None.
        class_weights (bool, optional): Whether to compute and apply class weights in the loss function.
                                        Defaults to False.
        device (torch.device, optional): Device to perform operations on. Defaults to torch.device("cuda").
    
    Returns:
        Tuple[nn.Module, nn.Module, nn.Module, torch.optim.Optimizer]: A tuple containing:
            - teacher_model: The teacher model loaded with fine-tuned weights.
            - student_model: The student model prepared for quantization-aware training.
            - criterion: The configured loss function.
            - optimizer: The configured optimizer for the student model.
    """
    # Determine the number of classes from the dataloader.
    num_classes = len(dataloader.dataset.classes)

    # Load the student model (as a standard model) using the provided student model name.
    student_model = setup_model(student, None, num_classes)
    
    if quant_mode:
        # Obtain example inputs from the dataloader for QAT preparation.
        example_inputs = next(iter(dataloader))[0].to(device)
        # Prepare the student model for QAT using the specified quantization mode.
        student_model = quantization_mode(student_model.to(device), quant_mode, example_inputs=(example_inputs,), config=config)

    # Load the teacher model (with fine-tuned weights) using the provided teacher model name and weights.
    teacher_model = setup_model(teacher, teacher_model_weights, num_classes)

    # Configure the loss function with optional class weights.
    criterion = setup_criterion(criterion_name, dataloader, class_weights)
    # Configure the optimizer for the student model.
    optimizer = setup_optimizer(student_model, optimizer_name, learning_rate)

    return teacher_model, student_model, criterion, optimizer