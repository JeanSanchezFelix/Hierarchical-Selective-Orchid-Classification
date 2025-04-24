import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datasets.registry import DATASET_REGISTRY
from model_compression.src.quantization.utils.inspect import is_quantized_model
from model_compression.src.utils.model_setup import setup_model
from model_compression.src.utils.metrics import calculate_metrics, plot_metric_bar, plot_confusion_matrix, plot_roc_auc_curve, plot_radar_chart

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

def evaluate(model_path: str, metadata_path: str, img_size: int, dataset: str, save_dir: str = None):
    """
    Perform inference on the test dataset and evaluate model performance.

    Args:
        model_path (str): Path to the saved model file.
        metadata_path (str): Path to the metadata of saved model file.
        dataset (str): The name of the dataset that will be used for evaluation.
        save_dir (str): Directory to save plots. If None, plots will not be saved.
    """
    # Set up device: use GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the saved model and its parameters
    model_data = torch.load(model_path, weights_only=True, map_location=device)
    metadata = torch.load(metadata_path)
    criterion = metadata["criterion"]
    batch_size = metadata["batch_size"]

    # Image normalization parameters (standard for ImageNet-pretrained models)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Define preprocessing transformations for the test dataset
    basic_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),        # Resize images to fixed size
        transforms.ToTensor(),                          # Convert images to PyTorch tensors
        transforms.Normalize(mean=mean, std=std)        # Normalize using ImageNet stats
    ])

    # Load the test dataset using a registry (assuming DATASET_REGISTRY exists)
    test_dataset = DATASET_REGISTRY[dataset](mode='test', transform=basic_transforms)
    test_loader = DataLoader(test_dataset.dataset, batch_size=batch_size, shuffle=False)

    # Retrieve class labels from the dataset
    class_labels = test_loader.dataset.classes
    
    filename = model_path.split('/')[-1]                # Split the path by '/' and get the last component (filename)
    model_name = filename.split('_best_model.pth')[0]   # Split the filename by '_best_model.pth' to extract the model name

    # Initialize and load the model
    model = setup_model(model_name, pretrained_weights=False, num_classes=len(class_labels))
    model.load_state_dict(model_data)
    model.to(device)  # Move the model to the same device as the inputs

    test_inference(model, test_loader, device, criterion, save_dir)

def compute_loss_and_predictions(outputs: torch.Tensor,
                                labels: torch.Tensor,
                                criterion: nn.Module
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes loss and generates predictions using the provided loss function.

    For loss functions like BCE/BCELoss, CrossEntropyLoss, or NLLLoss, the appropriate transformation 
    is applied to the raw outputs to compute both the loss and predicted class labels.

    Args:
        outputs (torch.Tensor): Raw model outputs (logits).
        labels (torch.Tensor): Ground truth labels.
        criterion (nn.Module): Loss function to use.

    Returns:
        Tuple containing:
            - loss (torch.Tensor): Computed loss.
            - probs (torch.Tensor): Predicted probabilities.
            - preds (torch.Tensor): Final predicted class labels.
    
    Raises:
        ValueError: If an unsupported loss function is provided.
    """
    if isinstance(criterion, nn.BCELoss):
        # BCELoss expects probabilities.
        probs = torch.sigmoid(outputs)
        loss = criterion(probs, labels.float().unsqueeze(1))
        preds = (probs >= 0.5).float()

    elif isinstance(criterion, nn.BCEWithLogitsLoss):
        # BCEWithLogitsLoss expects raw logits.
        loss = criterion(outputs, labels.float().unsqueeze(1))
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()

    elif isinstance(criterion, nn.CrossEntropyLoss):
        # CrossEntropyLoss expects raw logits and integer labels.
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, dim=1)

    elif isinstance(criterion, nn.NLLLoss):
        # NLLLoss expects log-probabilities.
        outputs = torch.log_softmax(outputs, dim=1)
        loss = criterion(outputs, labels)
        probs = torch.exp(outputs)
        _, preds = torch.max(outputs, dim=1)
    else:
        raise ValueError(f"Unsupported loss function: {type(criterion)}")

    return loss, probs, preds

def _compute_predictions(outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes predicted probabilities and class labels solely based on model outputs.

    Determines whether the task is binary or multiclass based on the shape of the outputs.

    Args:
        outputs (torch.Tensor): Raw model outputs (logits).

    Returns:
        Tuple containing:
            - probs (torch.Tensor): Predicted probabilities.
            - preds (torch.Tensor): Final predicted class labels.
    """
    # Determine if outputs correspond to binary classification.
    is_binary = (outputs.ndim == 1) or (outputs.shape[1] == 1)

    if is_binary:
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
    else:
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, dim=1)
    return probs, preds

def test_inference(model: nn.Module, data_loader: DataLoader, device, criterion=None, save_dir=None) -> dict | None:
    """
    Performs inference on a dataset, computes metrics, and optionally saves plots.

    Args:
        model (nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to perform inference on (CPU or GPU).
        save_dir (str): Directory to save plots. If None, plots will not be saved.
        quant (bool): Flag to indicate if the model is quantized. Default is False.
    """
    logging.info("Starting inference...")
    
    # Set model to evaluation mode.
    if is_quantized_model(model):
        torch.ao.quantization.allow_exported_model_train_eval(model)  # restores eval/train to call move_exported_model_* under the hood
    
    model.eval()

    running_loss, total_samples = 0.0, 0
    predictions, ground_truths, probabilities = [], [], []
    class_labels = data_loader.dataset.classes

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in tqdm(data_loader, desc="Inference Progress"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels).item() if criterion else 0.0
            probs, preds = _compute_predictions(outputs)

            batch_size = inputs.size(0)
            running_loss += loss * batch_size
            total_samples += batch_size

            ground_truths.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    logging.info("Inference complete.")   

    # Calculate metrics.
    y_true, y_pred = np.array(ground_truths), np.array(predictions)
    y_proba = np.array(probabilities) if probabilities else None
    metrics = calculate_metrics(y_true, y_pred, y_proba)

    if criterion:
        metrics["loss"] = running_loss / total_samples

    # Generate and save plots if save_dir is provided.
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_metric_bar(metrics, save_path=os.path.join(save_dir, "metrics_bar_chart.png"))
        if class_labels is not None:
            plot_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                labels=class_labels,
                save_path=os.path.join(save_dir, "confusion_matrix.png")
            )
        if y_proba is not None:
            plot_roc_auc_curve(y_true, y_proba, save_path=os.path.join(save_dir, "roc_auc_curve.png"))
        plot_radar_chart(metrics, save_path=os.path.join(save_dir, "radar_chart.png"))
        logging.info(f"Plots saved to {save_dir}")
    
    return metrics

def test_inference_onnx(
    onnx_model_path: str,
    test_loader: DataLoader,
    device: torch.device,  # Mostly will be CPU for ONNX
    save_dir: str = None
) -> None:
    """
    Performs inference on the test dataset using an ONNX model, computes metrics, and optionally saves plots.
    
    This function uses ONNX Runtime to run inference on the exported ONNX model.
    
    Args:
        onnx_model_path (str): Path to the exported ONNX model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to use for inference (typically CPU for ONNX).
        save_dir (str): Directory to save plots. If None, plots will not be saved.
    """
    # Load the ONNX model with ONNX Runtime.
    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    
    predictions, ground_truths, probabilities = [], [], []
    class_labels = test_loader.dataset.classes

    for inputs, labels in tqdm(test_loader, desc="ONNX Inference Progress"):
        # Convert input to NumPy array (and ensure it's on CPU).
        np_inputs = inputs.cpu().numpy()
        # Run inference using ONNX Runtime.
        ort_outs = session.run(None, {input_name: np_inputs})
        # Assume the first output is the prediction.
        outputs = ort_outs[0]
        # If needed, convert outputs to torch tensors for further processing.
        outputs_tensor = torch.tensor(outputs)
        # Compute predictions (this function must work with tensor or can be modified accordingly).
        probs, preds = compute_predictions(outputs_tensor)
        ground_truths.extend(labels.cpu().numpy())
        predictions.extend(preds.cpu().numpy())
        probabilities.extend(probs.cpu().numpy())

    # Compute metrics similar to the original function.
    y_true = np.array(ground_truths)
    y_pred = np.array(predictions)
    y_proba = np.array(probabilities) if probabilities else None
    metrics = calculate_metrics(y_true, y_pred, y_proba)

    print("ONNX Metrics Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_metric_bar(metrics, save_path=os.path.join(save_dir, "onnx_metrics_bar_chart.png"))
        if class_labels is not None:
            plot_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                labels=class_labels,
                save_path=os.path.join(save_dir, "onnx_confusion_matrix.png")
            )
        if y_proba is not None:
            plot_roc_auc_curve(y_true, y_proba, save_path=os.path.join(save_dir, "onnx_roc_auc_curve.png"))
        plot_radar_chart(metrics, save_path=os.path.join(save_dir, "onnx_radar_chart.png"))
        logging.info(f"ONNX plots saved to {save_dir}")

def test_inference_savedmodel(
    saved_model_path: str,
    data_dir: str,
    batch_size: int = 32,
    image_size: tuple = (224, 224),
    save_dir: str = None
) -> None:
    """
    Performs inference on the test dataset using a TensorFlow SavedModel,
    computes metrics, and optionally saves plots.

    The test dataset is loaded from a directory structure containing "train", "val", and "test" folders.
    It assumes that the test data is in the "test" folder within data_dir.

    Args:
        saved_model_path (str): Path to the TensorFlow SavedModel directory.
        data_dir (str): Root directory containing "train", "val", and "test" subdirectories.
        batch_size (int): Batch size for the test dataset. Defaults to 32.
        image_size (tuple): Desired image size (height, width). Defaults to (224, 224).
        save_dir (str): Directory to save plots. If None, plots will not be saved.
    """
    logging.info("Loading SavedModel for inference...")
    # Load the SavedModel; note that this does not return a Keras model with predict()
    model = tf.saved_model.load(saved_model_path)
    
    # Get the serving signature; this is a callable that accepts a dict of inputs.
    infer = model.signatures["serving_default"]
    
    # Determine the input key from the signature.
    input_key = list(infer.structured_input_signature[1].keys())[0]
    logging.info(f"Model input key: {input_key}")
    
    # Load the test dataset from the "test" folder.
    test_dir = os.path.join(data_dir, "test")
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False  # For reproducible order.
    )
    class_labels = test_dataset.class_names
    logging.info(f"Found classes: {class_labels}")
    
    # Preprocessing: 
    # 1. Rescale images from [0, 255] to [0, 1].
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    # 2. Transpose images from channels-last (H, W, C) to channels-first (C, H, W).
    # 3. Normalize using the same mean and std as during training.
    test_dataset = test_dataset.map(lambda x, y: (
        normalize_channels_first(tf.transpose(normalization_layer(x), perm=[0, 3, 1, 2])),
        y
    ))
    
    predictions_list = []
    ground_truths_list = []
    
    logging.info("Running inference on the test dataset using the serving signature...")
    # Iterate over the dataset and call the signature on each batch.
    for inputs, labels in tqdm(test_dataset, desc="TF Inference Progress"):
    # for inputs, labels in test_dataset:
        # Call the model's serving signature with the proper input key.
        outputs = infer(**{input_key: inputs})
        # Assume the first output is the one we need.
        output_key = list(outputs.keys())[0]
        batch_predictions = outputs[output_key]
        predictions_list.append(batch_predictions.numpy())
        ground_truths_list.append(labels.numpy())
    
    # Concatenate all predictions and ground truths.
    predictions = np.concatenate(predictions_list, axis=0)
    ground_truths = np.concatenate(ground_truths_list, axis=0)

    # Process predictions depending on the number of classes.
    if len(class_labels) == 2:
        # For binary classification: apply sigmoid and threshold at 0.5.
        probs = tf.nn.sigmoid(predictions).numpy()
        pred_labels = (probs >= 0.5).astype(int).flatten()
    else:
        # For multi-class classification: apply softmax and use argmax.
        probs = tf.nn.softmax(predictions, axis=-1).numpy()
        pred_labels = np.argmax(probs, axis=-1)
    
    # Calculate metrics using your helper function.
    metrics: dict[str, float] = calculate_metrics(ground_truths, pred_labels, probs)
    
    # Print the metrics.
    print("Metrics Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Generate and save plots if a save directory is provided.
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_metric_bar(metrics, save_path=os.path.join(save_dir, "metrics_bar_chart.png"))
        if class_labels:
            plot_confusion_matrix(
                y_true=ground_truths,
                y_pred=pred_labels,
                labels=class_labels,
                save_path=os.path.join(save_dir, "confusion_matrix.png")
            )
        if probs is not None:
            plot_roc_auc_curve(ground_truths, probs, save_path=os.path.join(save_dir, "roc_auc_curve.png"))
        plot_radar_chart(metrics, save_path=os.path.join(save_dir, "radar_chart.png"))
        logging.info(f"Plots saved to {save_dir}")

def normalize_channels_first(x: tf.Tensor) -> tf.Tensor:
    """
    Normalizes a tensor in channels-first format using the ImageNet mean and std.
    
    Args:
        x (tf.Tensor): Input tensor of shape (batch, 3, height, width) with values in [0, 1].
    
    Returns:
        tf.Tensor: Normalized tensor.
    """
    # Define the mean and std as constants.
    mean = tf.constant([0.485, 0.456, 0.406], shape=(3, 1, 1), dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], shape=(3, 1, 1), dtype=tf.float32)
    return (x - mean) / std