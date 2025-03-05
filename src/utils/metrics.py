import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
# Use a colorblind-friendly palette
sns.set_palette("colorblind")

def calculate_metrics(y_true, y_pred, y_proba=None) -> dict[str,float]:
    """
    Calculates classification metrics, supporting both binary and multiclass classification.

    Parameters:
        y_true (list or ndarray): True labels.
        y_pred (list or ndarray): Predicted labels.
        y_proba (ndarray, optional): Predicted probabilities or scores for all classes 
                                     (for AUC-Score in multiclass classification).

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    # Determine if the problem is binary or multiclass
    num_classes = len(set(y_true))

    # Choose averaging method
    average = 'binary' if num_classes == 2 else 'macro'

    # Compute metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, average=average),
        "Precision": precision_score(y_true, y_pred, average=average),
        "F1-Score": f1_score(y_true, y_pred, average=average),
    }

    # Compute AUC-Score
    if y_proba is not None:
        if num_classes == 2:
            metrics["AUC-Score"] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            # Multiclass AUC-Score (requires one-vs-rest probabilities)
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
    # plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=None, title: str = "Confusion Matrix", save_path: str = None):
    """
    Plots a confusion matrix heatmap.
    
    Args:
        y_true (list or ndarray): True labels.
        y_pred (list or ndarray): Predicted labels.
        labels (list): List of class labels.
        title (str): Title of the plot.
        save_path (str): Optional path to save the plot.
    """
        # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize="true")  # Normalize values
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot confusion matrix with adjusted text size
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, 
        yticklabels=labels, ax=ax, annot_kws={"size": 10}  # Set font size inside squares
    )
    
    # Adjust x-axis and y-axis tick positions to align labels with heatmap squares
    ax.set_xticks(np.arange(len(labels)) + 0.5)  
    ax.set_yticks(np.arange(len(labels)) + 0.5)  

    # Properly align x and y labels in the middle of the squares
    ax.set_xticklabels(labels, rotation=45, ha="center")  # Center horizontally
    ax.set_yticklabels(labels, rotation=45, va="center")   # Center vertically

    # Adjust tick parameters to move x-axis labels slightly down
    ax.xaxis.set_tick_params(length=0, labelsize=10, pad=10)  # Moves labels down a bit
    ax.yaxis.set_tick_params(length=0, labelsize=10)

    # Increase font size of axis labels
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Predicted Labels", fontsize=18, labelpad=10)  # Add padding to move label
    ax.set_ylabel("True Labels", fontsize=18)

    # Ensure the layout prevents label cutoff
    plt.tight_layout()

    # Save or show plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    # plt.show()


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
    # plt.show()

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
    # plt.show()

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
    # plt.show()

