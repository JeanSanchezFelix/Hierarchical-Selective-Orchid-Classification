import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
    matthews_corrcoef,
    log_loss
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# Use a colorblind-friendly palette
sns.set_palette("colorblind")

def calculate_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification performance metrics for binary or multiclass tasks.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        y_proba: Optional array of predicted probabilities or scores.

    Returns:
        A dict mapping metric names to their values.
    """
    # Determine whether the task is binary or multiclass.
    num_classes = len(np.unique(y_true))
    average = 'binary' if num_classes == 2 else 'macro'

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, average=average),
        "Precision": precision_score(y_true, y_pred, average=average),
        "F1-Score": f1_score(y_true, y_pred, average=average),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

    if y_proba is not None:
        if num_classes == 2:
            # For binary, assume probabilities for the positive class are in column index 1.
            metrics["AUC-Score"] = roc_auc_score(y_true, y_proba[:, 0])
        else:
            # For multiclass, compute the AUC using one-vs-rest probabilities.
            metrics["AUC-Score"] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')

    return metrics

def plot_metric_bar(
    metrics: Dict[str, float],
    title: str = 'Metrics',
    save_path: Optional[str] = None
) -> None:
    """
    Plot a bar chart of scalar metric values.

    Args:
        metrics: Dict of metric names to scalar values.
        title: Plot title.
        save_path: Path to save the figure.

    Raises:
        ValueError: If no valid metrics to plot.
    """
    # Filter out non-scalar metrics (e.g., confusion matrix)
    valid_metrics = {k: v for k, v in metrics.items() if np.isscalar(v)}
    if not valid_metrics:
        raise ValueError('No scalar metrics to plot.')

    # Extract keys and values for plotting
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    # Bar plot
    plt.figure(figsize=(10, 8))
    color_palette = sns.color_palette("Blues", len(metric_names))  # Different shades of blue
    sns.barplot(x=metric_names, y=metric_values, palette=color_palette, hue=metric_names, legend=False)

    # Add value labels on top of each bar
    for i, value in enumerate(metric_values):
        plt.text(
            i,               # Position on the X-axis (center of the bar)
            value + 0.01,    # Slightly above the bar
            f"{value:.2f}",  # Format the value
            ha='center',     # Align horizontally to the center of the bar
            color='black', fontsize=10
        )

    # Customize the X-axis and Y-axis
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title(title)
    plt.ylim(0, 1)  # Most metrics are within this range
    plt.yticks(np.arange(0, 1.1, 0.1))  # Increment Y-axis ticks by 0.1

    # Save or show plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def plot_confusion_matrix(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    labels: Optional[List[str]] = None,
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None
) -> None:
    """
    Plot a normalized confusion matrix heatmap.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        labels: Optional list of label names in order.
        title: Plot title.
        save_path: Path to save the figure.
    """
    # Compute confusion matrix with true normalization
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    
    # If labels not provided, extract unique labels
    if labels is None:
        labels = [str(l) for l in np.unique(np.concatenate([y_true, y_pred]))]
    
    # Create figure with increased width to accommodate labels
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with improved readability
    sns.heatmap(
        cm, 
        annot=True,  # Show numerical values
        fmt=".2f",   # Two decimal places
        cmap="Blues",  # Color scheme
        cbar_kws={'label': 'Normalized Frequency'},  # Colorbar label
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"fontsize": 12}  # Smaller font for annotations
    )
    
    # Adjust tick positions to center labels in squares
    plt.xticks(
        np.arange(len(labels)) + 0.5,  # Center labels in squares
        labels, 
        rotation=45,  # 45-degree angle
        ha='right',   # Horizontal alignment
        rotation_mode='anchor'  # Ensures rotation is applied from the right
    )
    
    plt.yticks(
        np.arange(len(labels)) + 0.5,  # Center labels in squares
        labels, 
        rotation=45,  # 45-degree angle
        ha='right',   # Horizontal alignment
        rotation_mode='anchor'  # Ensures rotation is applied from the right
    )
    
    # Set title and axis labels
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Predicted Labels", fontsize=12, labelpad=10)
    plt.ylabel("True Labels", fontsize=12, labelpad=10)
    
    # Ensure layout is tight to prevent label cutoff
    plt.tight_layout()
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()


def plot_train_val_curve(
    history: Dict[str, List[float]],
    metric_name: str = 'loss',
    title: str = 'Training vs Validation',
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation metric curves over epochs.

    Args:
        history: Dict with keys 'train' and 'val' lists of values.
        metric_name: Label for the Y-axis.
        title: Plot title.
        save_path: Path to save the figure.
    """
    
    plt.figure(figsize=(10, 8))
    epochs = range(1, len(history['train']) + 1)

    # Line plot for training and validation
    plt.plot(epochs, history['train'], label=f"Training {metric_name}", color='blue')
    plt.plot(epochs, history['val'], label=f"Validation {metric_name}", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()

    # Save or show plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def plot_roc_auc_curve(
    y_true: Union[List[int], np.ndarray],
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = 'ROC AUC Curve',
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curve for binary or multiclass classification.

    Args:
        y_true: True class labels.
        y_proba: Predicted probabilities array of shape (n_samples, n_classes).
        class_names: Optional list of class names corresponding to integer labels.
            If provided, length must equal number of unique classes. Defaults to None.
        title: Plot title.
        save_path: Path to save the figure.
    """
    # Determine if the problem is binary or multiclass
    classes = np.unique(y_true)
    num_classes = len(classes)

    # Validate class_names
    if class_names is not None:
        if len(class_names) != num_classes:
            raise ValueError(
                f"Length of class_names ({len(class_names)}) must match number of classes ({num_classes})"
            )
    else:
        class_names = [str(cls) for cls in classes]

    spectrum = plt.cm.nipy_spectral(np.linspace(0, 1, num_classes))
    colors = list(spectrum)

    plt.figure(figsize=(10, 8))

    # Binary classification
    if num_classes == 2:
        # Check proba shape
        if y_proba.shape[1] != 2:
            raise ValueError(f"Expected y_proba with shape (*, 2) for binary, got {y_proba.shape}")
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr,
            color=colors[1],
            lw=2,
            llabel=f"{class_names[1]} (AUC = {roc_auc:.2f})"
        )
    else:
        # Multiclass: binarize and plot per class
        y_true_bin = label_binarize(y_true, classes=classes)
        if y_proba.shape[1] != num_classes:
            raise ValueError(
                f"Expected y_proba with shape (*, {num_classes}), got {y_proba.shape}"
            )
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, tpr,
                color=colors[i],
                lw=2,
                label=f"{class_names[i]} (AUC = {roc_auc:.2f})"
            )

    # Plot formatting
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=2)  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")

    # Save or show plot
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

def plot_calibration_curve(
    y_true: Union[List[int], np.ndarray],
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    save_path: Optional[str] = None):
    """
    Plots the calibration (reliability) curve.

    Args:
        y_true: Ground truth binary labels.
        y_proba: Predicted probabilities (assumes binary classification).
        class_names: Optional list of class names corresponding to integer labels.
            If provided, length must equal number of unique classes. Defaults to None.
        n_bins: Number of bins to divide probability range [0, 1].
        title: Title of the plot.
        save_path: Optional path to save the plot.
    """
    classes = np.unique(y_true)
    num_classes = len(classes)

    # Validate class_names
    if class_names is not None:
        if len(class_names) != num_classes:
            raise ValueError(
                f"Length of class_names ({len(class_names)}) must match number of classes ({num_classes})"
            )
    else:
        class_names = [str(cls) for cls in classes]

    spectrum = plt.cm.nipy_spectral(np.linspace(0, 1, num_classes))
    colors = list(spectrum)

    # Define marker and line style cycles to ensure distinct shapes
    markers = ['o', 's', 'D', 'v', '^', '<', '>', 'p', '*', 'h', '+', 'x', 'd', '|', '_']


    plt.figure(figsize=(10, 8))

    # Binary calibration
    if y_proba.shape[1] == 2 and num_classes == 2:
        prob_true, prob_pred = calibration_curve(y_true, y_proba[:, 1], n_bins=n_bins)
        plt.plot(
            prob_pred, prob_true,
            marker=markers[0],
            linestyle='-',
            lw=2,
            label=class_names[i],
            color=colors[i]
        )
    else:
        # Multiclass one-vs-rest calibration
        y_true_bin = label_binarize(y_true, classes=classes)
        if y_proba.shape[1] != num_classes:
            raise ValueError(
                f"Expected y_proba with shape (*, {num_classes}), got {y_proba.shape}"
            )
        for i in range(num_classes):
            prob_true, prob_pred = calibration_curve(
                y_true_bin[:, i], y_proba[:, i], n_bins=n_bins
            )
            plt.plot(
                prob_pred, prob_true,
                marker=markers[i % len(markers)],
                linestyle='-',
                lw=2,
                label=class_names[i],
                color=colors[i]
            )

    # Plot reference line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
        
def plot_log_loss(
    history: Dict[str, List[float]],
    title: str ="Log Loss Over Epochs",
    save_path: Optional[str] =None
) -> None:
    """
    Plots training and validation log loss over epochs.

    Args:
        history: A dictionary with keys 'train' and 'val', each containing a list of log loss values.
        title: Title of the plot.
        save_path: Optional path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    epochs = range(1, len(history['train']) + 1)
    plt.plot(epochs, history['train'], label='Training Log Loss', color='blue')
    plt.plot(epochs, history['val'], label='Validation Log Loss', color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Log Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def plot_radar_chart(
    history: Dict[str, List[float]],
    title: str ="Radar Chart for Metrics", 
    save_path: str =None
) -> None:
    """
    Generates a radar plot based on metrics.

    Args:
        metrics: Dictionary of metric names and their scalar values (excluding Confusion Matrix).
        title: Title of the plot.
        save_path: Optional path to save the plot.
    """
    # Filter out non-scalar metrics (e.g., confusion matrix)
    metrics = {k: v for k, v in history.items() if np.isscalar(v)}
    
    # If no valid metrics remain, raise an error
    if not metrics:
        raise ValueError("No valid scalar metrics to plot.")
    
    # Extract keys and values for plotting
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    # Radar plot requires a closed loop
    metric_names.append(metric_names[0])
    metric_values.append(metric_values[0])

    # Calculate angles for the radar plot
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=True)

    # Create radar plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, metric_values, color='blue', alpha=0.3)
    ax.plot(angles, metric_values, color='blue', linewidth=2)

    # Add labels and title
    ax.set_yticks(np.arange(0.1, 1.1, 0.1))  # Y-axis increments by 0.1
    ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0.1, 1.1, 0.1)], color='gray')
    ax.set_xticks(angles)  # Use all angles, including the closing one
    ax.set_xticklabels(metric_names, fontsize=10, rotation=30, ha='right')  # Rotate labels for better alignment
    ax.set_rlabel_position(180 / len(metric_names))  # Offset radial labels slightly

    # Add title with padding
    ax.set_title(title, size=16, pad=30)

    # Adjust layout to avoid title overlap
    plt.subplots_adjust(top=0.85)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

