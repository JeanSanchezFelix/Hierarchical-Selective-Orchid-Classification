import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_top_k_accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> list:
    """
    Computes the top-k accuracy for the specified values of k.
    
    Parameters:
        output (torch.Tensor): Model outputs.
        target (torch.Tensor): Ground truth labels.
        topk (tuple): Tuple of top-k values to compute.
    
    Returns:
        list: Accuracy percentages for each k in topk.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.item() * 100.0 / batch_size))
        return res
    
def benchmark_model_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module
) -> dict[str, float]:
    """
    Benchmarks the given model by computing performance metrics and timing statistics.
    It runs the model over the evaluation data and computes accuracy, precision, recall,
    F1-Score, average inference time per sample, and throughput (samples per second).

    Parameters:
        model (nn.Module): The model to benchmark.
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        criterion (nn.Module): The loss function used (for reporting loss, if needed).

    Returns:
        dict[str, float]: A dictionary containing the computed metrics.
    """
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_samples = 0
    
    # Start timing the inference
    start_time = time.time()
    
    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images, targets  # Assume models are on CPU for quantization
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            # Get predicted classes (assuming classification with softmax/logits)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    end_time = time.time()
    total_time = end_time - start_time

    # Compute average loss
    avg_loss = total_loss / total_samples

    # Compute standard classification metrics
    accuracy = accuracy_score(all_targets, all_preds) * 100.0
    precision = precision_score(all_targets, all_preds, average="macro") * 100.0
    recall = recall_score(all_targets, all_preds, average="macro") * 100.0
    f1 = f1_score(all_targets, all_preds, average="macro") * 100.0
    
    # Compute average inference time per sample and throughput
    avg_inference_time = total_time / total_samples
    throughput = total_samples / total_time

    metrics = {
        "Average Loss": avg_loss,
        "Accuracy (%)": accuracy,
        "Precision (%)": precision,
        "Recall (%)": recall,
        "F1 Score (%)": f1,
        "Avg Inference Time (s/sample)": avg_inference_time,
        "Throughput (samples/s)": throughput,
    }
    
    # Print the benchmark results
    print("Benchmark Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

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