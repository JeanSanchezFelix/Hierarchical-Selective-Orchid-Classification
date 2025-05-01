import torch
import torch.nn as nn
from typing import Tuple

def _compute_loss_and_predictions(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute loss, predicted probabilities, and predicted labels given model outputs and a loss function.

    Supports BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, and NLLLoss.

    Args:
        outputs: Raw model outputs (logits or probabilities).
        labels: Ground-truth labels (integer class indices or binary labels).
        criterion: A PyTorch loss module.

    Returns:
        A tuple of (loss, probabilities, predictions):
        - loss: Scalar loss tensor.
        - probabilities: Tensor of predicted probabilities.
        - predictions: Tensor of class predictions.

    Raises:
        ValueError: If the criterion is not supported.
    """
    # Binary classification with BCE
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
    # Multi-class classification
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
        raise ValueError(f"Unsupported loss function: {criterion.__class__.__name__}")

    return loss, probs, preds

def _compute_predictions(
    outputs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Derive predicted probabilities and labels from raw outputs for binary or multi-class tasks.

    Args:
        outputs: Raw model outputs (logits).

    Returns:
        A tuple of (probabilities, predictions):
        - probabilities: Tensor of predicted probabilities.
        - predictions: Tensor of class predictions.
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