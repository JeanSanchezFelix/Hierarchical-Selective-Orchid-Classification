import torch
import torch.nn as nn

def _compute_loss_and_predictions(outputs: torch.Tensor,
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