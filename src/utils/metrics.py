import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from typing import Optional, Union

# Use a colorblind-friendly palette
sns.set_palette("colorblind")

def calculate_metrics(y_true, y_pred, y_proba = None) -> dict[str, float]:
    """
    Calculates classification metrics, supporting both binary and multiclass classification.

    Parameters:
        y_true (list or ndarray): True labels.
        y_pred (list or ndarray): Predicted labels.
        y_proba (ndarray, optional): Predicted probabilities or scores for all classes 
                                     (required for AUC-Score computation).

    Returns:
        dict[str, float]: A dictionary containing the computed metrics.
    """
    # Determine whether the task is binary or multiclass.
    num_classes = len(set(y_true))
    average = 'binary' if num_classes == 2 else 'macro'

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, average=average),
        "Precision": precision_score(y_true, y_pred, average=average),
        "F1-Score": f1_score(y_true, y_pred, average=average),
    }

    if y_proba is not None:
        if num_classes == 2:
            # For binary, assume probabilities for the positive class are in column index 1.
            metrics["AUC-Score"] = roc_auc_score(y_true, y_proba[:, 0])
        else:
            # For multiclass, compute the AUC using one-vs-rest probabilities.
            metrics["AUC-Score"] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')

    return metrics

def plot_metric_bar(metrics: dict, title: str = "Performance Metrics", save_path: str = None):
    """
    Plots a bar chart for given metrics.
    
    Args:
        metrics (dict): Dictionary of metric names (keys) and their values.
        title (str): Title of the plot.
        save_path (str): Optional path to save the plot.
    """
    # Filter out non-scalar metrics (e.g., confusion matrix)
    metrics = {k: v for k, v in metrics.items() if np.isscalar(v)}
    
    # If no valid metrics remain, raise an error
    if not metrics:
        raise ValueError("No valid scalar metrics to plot.")

    # Extract keys and values for plotting
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    # Bar plot
    plt.figure(figsize=(10, 6))
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
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def plot_confusion_matrix(
    y_true: Union[list, np.ndarray], 
    y_pred: Union[list, np.ndarray], 
    labels: Optional[list[str]] = None, 
    title: str = "Confusion Matrix", 
    save_path: Optional[str] = None
) -> None:
    """
    Plots a confusion matrix heatmap with improved label readability.
    
    Args:
        y_true (list or ndarray): True labels for the classification task.
        y_pred(list or ndarray): Predicted labels for the classification task.
        labels (list): Optional list of class labels. If None, unique labels will be extracted.
        title (str): Title of the confusion matrix plot.
        save_path (str): Optional file path to save the plot.
    """
    # Compute confusion matrix with true normalization
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    
    # If labels not provided, extract unique labels
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred])).tolist()
    
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


def plot_train_val_curve(metrics: dict, metric_name: str = "Loss", 
                         title: str = "Training vs Validation", save_path: str = None):
    """
    Plots training and validation metrics over epochs.
    
    Args:
        metrics (dict): Dictionary of training and validation metric values per epoch.
        metric_name (str): Name of the metric being plotted.
        title (str): Title of the plot.
        save_path (str): Optional path to save the plot.
    """
    
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(metrics['train']) + 1)

    # Line plot for training and validation
    plt.plot(epochs, metrics['train'], label=f"Training {metric_name}", color='blue')
    plt.plot(epochs, metrics['val'], label=f"Validation {metric_name}", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()

    # Save or show plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def plot_roc_auc_curve(y_true, y_proba, title="ROC-AUC Curve", save_path=None):
    """
    Plots the ROC-AUC curve for binary or multiclass classification.

    Args:
        y_true (array-like): True labels.
        y_proba (array-like): Predicted probabilities for each class.
        title (str): Title of the plot.
        save_path (str, optional): Path to save the plot.

    """

    # Determine if the problem is binary or multiclass
    num_classes = len(set(y_true))

    plt.figure(figsize=(10, 6))

    # Binary Classification
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])  # Get FPR and TPR
        roc_auc = auc(fpr, tpr)  # Compute AUC
        plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")

    # Multiclass Classification
    else:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))  # Convert labels to one-hot
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC = {roc_auc:.2f})")

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

def plot_radar_chart(metrics, title="Radar Chart for Metrics", save_path=None):
    """
    Generates a radar plot based on metrics.

    Args:
        metrics (dict): Dictionary of metric names and their scalar values (excluding Confusion Matrix).
        title (str): Title of the plot.
        save_path (str): Optional path to save the plot.
    """
    # Filter out non-scalar metrics (e.g., confusion matrix)
    metrics = {k: v for k, v in metrics.items() if np.isscalar(v)}
    
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

