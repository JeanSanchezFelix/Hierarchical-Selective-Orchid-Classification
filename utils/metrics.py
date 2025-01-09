import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix

# TODO: TEST IF IT WORKS

# Use a colorblind-friendly palette
sns.set_palette("colorblind")
plt.style.use('seaborn-colorblind')

def calculate_metrics(y_true, y_pred, y_proba=None):
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
    is_binary = num_classes == 2

    # Choose averaging method
    average = 'binary' if is_binary else 'macro'

    # Compute metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, average=average),
        "Precision": precision_score(y_true, y_pred, average=average),
        "F1-Score": f1_score(y_true, y_pred, average=average),
    }

    # Compute AUC-Score
    if y_proba is not None:
        if is_binary:
            metrics["AUC-Score"] = roc_auc_score(y_true, y_proba)
        else:
            # Multiclass AUC-Score (requires one-vs-rest probabilities)
            metrics["AUC-Score"] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')

    # Confusion matrix
    metrics["Confusion Matrix"] = confusion_matrix(y_true, y_pred)

    return metrics

def plot_metric_bar(metrics: dict, title: str = "Performance Metrics", save_path: str = None):
    """
    Plots a bar chart for given metrics.
    
    Args:
        metrics (dict): Dictionary of metric names (keys) and their values.
        title (str): Title of the plot.
        save_path (str): Optional path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    # Bar plot
    sns.barplot(x=metric_values, y=metric_names)
    plt.xlabel("Score")
    plt.ylabel("Metrics")
    plt.title(title)
    plt.xlim(0, 1)  # Most metrics are within this range

    # Save or show plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(conf_matrix, labels=None, title: str = "Confusion Matrix", save_path: str = None):
    """
    Plots a confusion matrix heatmap.
    
    Args:
        conf_matrix (2D array): Confusion matrix data.
        labels (list): List of class labels.
        title (str): Title of the plot.
        save_path (str): Optional path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)

    # Save or show plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_train_val_curve(train_metrics: list, val_metrics: list, metric_name: str = "Accuracy", 
                         title: str = "Training vs Validation", save_path: str = None):
    """
    Plots training and validation metrics over epochs.
    
    Args:
        train_metrics (list): Training metric values per epoch.
        val_metrics (list): Validation metric values per epoch.
        metric_name (str): Name of the metric being plotted.
        title (str): Title of the plot.
        save_path (str): Optional path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_metrics) + 1)

    # Line plot for training and validation
    plt.plot(epochs, train_metrics, label=f"Training {metric_name}", color='blue')
    plt.plot(epochs, val_metrics, label=f"Validation {metric_name}", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()

    # Save or show plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_radar_chart(metrics, title="Radar Chart for Metrics", save_path=None):
    """
    Generates a radar plot based on metrics.

    Args:
        metrics (dict): Dictionary of metric names and their values (excluding Confusion Matrix).
        title (str): Title of the plot.
        save_path (str): Optional path to save the plot.
    """
    # Exclude Confusion Matrix and prepare data
    metric_names = [key for key in metrics.keys() if key != "Confusion Matrix"]
    metric_values = [metrics[key] for key in metric_names]

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
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color='gray')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_title(title, size=16)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()