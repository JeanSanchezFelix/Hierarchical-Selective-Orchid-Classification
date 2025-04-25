import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import ai_edge_torch

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from typing import Callable, Optional, Any, Tuple

# Import quantization functions from PyTorch.
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config

# Set up logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(stream_handler)

def model_size(model: nn.Module) -> float:
    """
    Computes the model size (state_dict size) in MB.

    Args:
        model (torch.nn.Module): The model whose state dict size will be measured.

    Returns:
        float: The model size in megabytes.
    """
    temp_pth_path = "temp.pth"
    try:
        torch.save(model.state_dict(), temp_pth_path)
        pth_size = os.path.getsize(temp_pth_path) / 1e6 if os.path.exists(temp_pth_path) else 0.0
        return pth_size
    finally:
        if os.path.exists(temp_pth_path):
            os.remove(temp_pth_path)

def is_quantized_model(model: torch.nn.Module) -> bool:
    """
    Checks if the model appears to be a quantized model by testing its type name.

    Args:
        model (torch.nn.Module): The model to check.

    Returns:
        bool: True if the model's type name is "GraphModule", indicating it is quantized.
    """
    return type(model).__name__ == "GraphModule"

def warm_up(model: nn.Module, dataloader: DataLoader, device: torch.device, num_warmup: int = 5) -> None:
    """
    Runs a warm-up phase to stabilize the model and device before benchmarking.

    Args:
        model (torch.nn.Module): The model to warm up.
        dataloader (DataLoader): DataLoader providing input data.
        device (torch.device): Device on which to run inference.
        num_warmup (int): Number of warm-up batches to run.
    """
    # Set the model to evaluation mode.
    if is_quantized_model(model):
        model = torch.ao.quantization.move_exported_model_to_eval(model)
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
) -> Tuple[float, float]:
    """
    Measures the average inference time per sample and throughput of a model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader providing inference data.
        device (torch.device): Device on which to run inference.
        num_warmup (int): Number of warm-up batches before timing.
        num_trials (int): Number of batches to use for timing inference.

    Returns:
        Tuple[float, float]: (average inference time per sample in seconds, throughput in samples/sec)
    """
    # Set model to evaluation mode.
    if is_quantized_model(model):
        model = torch.ao.quantization.move_exported_model_to_eval(model)
    else:
        model.eval()

    # Warm-up before timing.
    warm_up(model, dataloader, device, num_warmup=num_warmup)

    total_time = 0.0
    total_samples = len(dataloader.dataset)

    # Timing loop: Run a fixed number of batches.
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

    return avg_time_per_sample, throughput

def quantize_pytorch_model(model: nn.Module, quant_mode: str, save_dir: str) -> nn.Module:
    """
    Quantizes the model using either FX Graph Mode or Export (PT2E) Mode.
    Saves the quantized state dict to save_dir.

    Args:
        model (torch.nn.Module): The model to quantize.
        quant_mode (str): Quantization mode, "fx" or "export".
        save_dir (str): Path to save the quantized state dict.

    Returns:
        torch.nn.Module: The quantized model.
    """
    if quant_mode == "fx":
        logging.info("Quantizing model using FX Graph Mode.")
        quantized_model = convert_fx(model)
    elif quant_mode == "export":
        logging.info("Quantizing model using PT2E Export Mode.")
        quantized_model = convert_pt2e(model, fold_quantize=False)
    else:
        raise ValueError("Invalid mode. Choose either 'fx' or 'export'.")
    
    torch.save(quantized_model.state_dict(), save_dir)
    return quantized_model

def quantization_mode(
    model: torch.nn.Module, 
    mode: str, 
    example_inputs: Optional[Tuple[torch.Tensor, ...]] = None, 
    config: str = None
) -> torch.nn.Module:
    """
    Prepares the model for quantization-aware training using either FX Graph Mode or Export Mode.

    Args:
        model (torch.nn.Module): The model to prepare.
        mode (str): The quantization mode: "fx" or "export".
        example_inputs (Optional[Tuple[torch.Tensor, ...]]): Example inputs required for the preparation.
        config (str): Configuration string for qconfig; if None, defaults to "qnnpack".

    Returns:
        torch.nn.Module: The prepared model.
    """
    if config:
        qconfig = torch.ao.quantization.get_default_qat_qconfig(config)
    else:
        qconfig = torch.ao.quantization.get_default_qat_qconfig("qnnpack")
    
    if mode == "fx":
        if example_inputs is None:
            raise ValueError("example_inputs is required for FX Graph Mode QAT.")
        qconfig_mapping = torch.ao.quantization.QConfigMapping().set_global(qconfig)
        model.qconfig = qconfig_mapping
        model = prepare_qat_fx(model, qconfig_mapping, example_inputs)
        print("Model prepared using FX Graph Mode QAT.")
    elif mode == "export":
        if example_inputs is None:
            raise ValueError("example_inputs is required for Export Mode QAT.")
        model = torch.export.export_for_training(model, example_inputs).module()
        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_qat=True))
        model = prepare_qat_pt2e(model, quantizer)
        print("Model prepared using Export Mode QAT.")
    else:
        raise ValueError("Invalid mode. Choose either 'eager', 'fx' or 'export'.")
    
    return model

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> dict[str, list[float]]:
    """
    Trains the model on the provided training and validation DataLoaders.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int): Number of training epochs.
        criterion (Callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device for training.

    Returns:
        dict[str, list[float]]: A dictionary with training and validation losses per epoch.
    """
    loss_dict = {'train': [], 'val': []}
    
    for epoch in range(epochs):
        for phase in ['train', 'val']:
            is_train = (phase == 'train')
            
            # Set the model mode appropriately.
            if is_train:
                torch.ao.quantization.move_exported_model_to_train(model)
            else:
                torch.ao.quantization.move_exported_model_to_eval(model)
            
            data_loader = train_loader if is_train else val_loader

            running_loss = 0.0
            total_samples = 0.0

            with tqdm(total=len(data_loader), desc=f"{phase.capitalize()} Epoch {epoch+1}/{epochs}") as pbar:
                for inputs, labels in data_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    if is_train:
                        optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(is_train):
                        model = model.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        if is_train:
                            loss.backward()
                            optimizer.step()
                    
                    batch_size = inputs.size(0)
                    running_loss += loss.item() * batch_size
                    total_samples += batch_size
                    pbar.set_postfix(loss=f"{running_loss/total_samples:.4f}")
                    pbar.update(1)
                    
            epoch_loss = running_loss / total_samples
            loss_dict[phase].append(epoch_loss)
            logging.info(f"{phase.capitalize()} Loss: {epoch_loss:.4f}")
    
    return loss_dict

def get_random_dataloader(num_samples: int, batch_size: int, input_shape: Tuple[int, int, int], num_classes: int) -> DataLoader:
    """
    Creates a DataLoader with random data for testing purposes.

    Args:
        num_samples (int): Number of samples in the dataset.
        batch_size (int): Batch size.
        input_shape (Tuple[int, int, int]): Shape of each input (C, H, W).
        num_classes (int): Number of classes.

    Returns:
        DataLoader: A DataLoader yielding random inputs and labels.
    """
    inputs = torch.randn(num_samples, *input_shape)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_qat(
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_dir: str,
    learning_rate: float = 0.001,
    epochs: int = 5,
    quant_mode: str = "export",
    config: str = "qnnpack",
    device: torch.device = torch.device("cpu")
) -> Tuple[nn.Module, nn.Module]:
    """
    Sets up a MobileNetV2 model for quantization-aware training on random data,
    trains it, and returns both the original and quantized models.

    Args:
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        save_dir (str): Directory to save quantized state dict.
        learning_rate (float): Learning rate.
        epochs (int): Number of training epochs.
        quant_mode (str): Quantization mode ("export" or "fx").
        config (str): Qconfig configuration string.
        device (torch.device): Device to use.

    Returns:
        Tuple[nn.Module, nn.Module]: (original model, quantized model)
    """
    # Load a pre-trained MobileNetV2.
    model = models.mobilenet_v2(weights='DEFAULT')

    num_features = model.classifier[1].in_features
    # Set model output layer: here using 14 classes as an example.
    model.classifier[1] = nn.Linear(num_features, 14)

    # Create example inputs for QAT preparation.
    example_inputs = torch.rand(1, 3, 224, 224).to(device)
    # Prepare the model for QAT.
    prepared_model = quantization_mode(model, quant_mode, example_inputs=(example_inputs,), config=config)
    
    optimizer_obj = torch.optim.Adam(prepared_model.parameters(), lr=learning_rate)
    criterion_fn = nn.CrossEntropyLoss()
    
    logging.info("Model setup complete.")
    os.makedirs(save_dir, exist_ok=True)
    
    logging.info("Starting training...")
    start_time = time.time()
    loss_dict = train(
        model=prepared_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        criterion=criterion_fn,
        optimizer=optimizer_obj,
        device=device
    )
    elapsed_time = time.time() - start_time
    logging.info(f"Training complete in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
    
    # Quantize the model for export.
    quantized_model = quantize_pytorch_model(prepared_model.to("cpu"), quant_mode, save_dir=os.path.join(save_dir, "quantized_state.pth"))
    
    return model, quantized_model

def main():
    learning_rate = 0.001
    epochs = 5
    save_dir = "models/Quantized"
    device = torch.device("cpu")
    os.makedirs(save_dir, exist_ok=True)

    # Create random dataloaders for training and validation.
    train_loader = get_random_dataloader(num_samples=128, batch_size=32, input_shape=(3, 224, 224), num_classes=14)
    val_loader   = get_random_dataloader(num_samples=64, batch_size=32, input_shape=(3, 224, 224), num_classes=14)
    
    # Train QAT on random data and get the models.
    original_model, quantized_model = train_qat(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=save_dir,
        learning_rate=learning_rate,
        epochs=epochs,
        quant_mode="export",
        config="qnnpack",
        device=device
    )
    
    # Report model sizes.
    original_size = model_size(original_model)
    quantized_size = model_size(quantized_model)
    logging.info(f"Original model size: {original_size:.4f} MB : Quantized model size: {quantized_size:.4f} MB")

    # # Measure inference performance.
    # original_inference_time, original_throughput = measure_inference_performance(original_model, val_loader, device)
    # quantized_inference_time, quantized_throughput = measure_inference_performance(quantized_model, val_loader, device)

    # logging.info(f"Original inference time: {original_inference_time:.6f} sec/sample")
    # logging.info(f"Quantized inference time: {quantized_inference_time:.6f} sec/sample")
    # logging.info(f"Original throughput: {original_throughput:.2f} samples/sec")
    # logging.info(f"Quantized throughput: {quantized_throughput:.2f} samples/sec")

    # example_inputs = next(iter(train_loader))[0].to("cpu")
        
    # logging.info("Converting PyTorch model to TensorFlow Lite format.")
    # # Ensure the model is in evaluation mode for export.
    # torch.ao.quantization.move_exported_model_to_eval(quantized_model)
    # # Convert the model using the ai_edge_torch conversion utility.
    # edge_model = ai_edge_torch.convert(quantized_model, (example_inputs,))
    # # Export the converted model to the specified directory.
    # tflite_dir = "tflite_test.tflite"
    # edge_model.export(tflite_dir)
    # logging.info(f"TFLite model exported to {tflite_dir}")
    # tflite_model_size = model_size(quantized_model)
    # logging.info(f"TfLite model size: {tflite_model_size:.4f} MB")

if __name__ == "__main__":
    main()
