import csv
import os
from pathlib import Path
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
from pycm import ConfusionMatrix

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
    # The model can have output columns for classes that do not appear in a
    # particular split.  Metrics must use the classes observed in ``y_true``,
    # rather than infer the number of score columns from that split.
    observed_classes = np.unique(y_true)
    num_classes = len(observed_classes)
    average = 'binary' if np.array_equal(observed_classes, np.array([0, 1])) else 'macro'

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average=average, zero_division=0),
        "MCC": 0.0 if num_classes < 2 else matthews_corrcoef(y_true, y_pred)
    }
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        if y_proba.ndim != 2:
            raise ValueError("y_proba must have shape (n_samples, n_classes).")
        missing_probability_column = observed_classes.size and (
            observed_classes[0] < 0 or observed_classes[-1] >= y_proba.shape[1]
        )
        if missing_probability_column and not (num_classes == 2 and y_proba.shape[1] == 1):
            raise ValueError("y_true contains a class without a corresponding y_proba column.")

        if num_classes < 2:
            # ROC AUC is undefined when a split contains only one class.
            metrics["AUC-Score"] = float("nan")
        elif num_classes == 2:
            # sklearn treats the greater label as the positive class.  Select
            # its actual model-output column; class IDs need not be consecutive.
            positive_proba = y_proba[:, 0] if y_proba.shape[1] == 1 else y_proba[:, observed_classes[1]]
            metrics["AUC-Score"] = roc_auc_score(y_true, positive_proba)
        else:
            # sklearn requires multiclass score rows to sum to one.  Remove
            # output columns for classes absent from this split, then
            # renormalize to evaluate only the observed one-vs-rest classes.
            observed_proba = y_proba[:, observed_classes]
            observed_proba = observed_proba / observed_proba.sum(axis=1, keepdims=True)
            metrics["AUC-Score"] = roc_auc_score(
                y_true,
                observed_proba,
                labels=observed_classes,
                multi_class='ovr',
                average='macro',
            )

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
    save_path: Optional[str] = None,
) -> None:
    """Plot a normalized confusion matrix through PyCM.

    PyCM receives display labels directly, preserving class names in the plot
    rather than relying on a separate tick-label mapping.
    """
    true = np.asarray(y_true).reshape(-1)
    predicted = np.asarray(y_pred).reshape(-1)
    class_ids = np.unique(np.concatenate((true, predicted))).astype(int)
    if labels is None:
        display_labels = [str(class_id) for class_id in class_ids]
    elif len(labels) == len(class_ids):
        display_labels = [str(label) for label in labels]
    elif class_ids.size and class_ids.min() >= 0 and class_ids.max() < len(labels):
        display_labels = [str(labels[class_id]) for class_id in class_ids]
    else:
        raise ValueError("labels must match the observed classes or be indexed by class ID.")
    label_map = dict(zip(class_ids, display_labels))
    actual_labels = [label_map[int(value)] for value in true]
    predicted_labels = [label_map[int(value)] for value in predicted]

    detailed_view = len(class_ids) <= 30
    cm = ConfusionMatrix(actual_vector=actual_labels, predict_vector=predicted_labels)
    axes = cm.plot(
        normalized=True,
        title=title,
        number_label=detailed_view,
        cmap=plt.cm.Blues,
        plot_lib="matplotlib",
    )
    figure = axes.figure
    figure.set_size_inches((10, 8) if detailed_view else (18, 16))
    if not detailed_view:
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title(f"{title} ({len(class_ids)} classes; labels in confusion_matrix.csv)")
    figure.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        figure.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(figure)
    else:
        plt.show()


def export_readable_metrics_report(
    metrics: Dict[str, float], y_true, y_pred, y_proba: Optional[np.ndarray],
    save_dir: str, class_names: Optional[List[str]] = None,
    image_paths: Optional[List[str]] = None,
) -> Path:
    """Export Markdown and CSV alternatives for unreadable large-class plots."""
    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    true, predicted = np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)
    if image_paths is not None and len(image_paths) != len(true):
        raise ValueError("image_paths must have one entry per prediction.")
    class_ids = np.unique(np.concatenate((true, predicted))).astype(int)
    counts = (np.array([[len(true)]], dtype=int) if len(class_ids) == 1
              else confusion_matrix(true, predicted, labels=class_ids))
    totals = counts.sum(axis=1)

    def label_for(class_id: int) -> str:
        if class_names is not None and 0 <= class_id < len(class_names):
            return str(class_names[class_id])
        return str(class_id)

    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "true_class_id", "true_class", "predicted_class_id", "predicted_class", "count", "true_class_rate", "true_image_paths",
        ])
        writer.writeheader()
        for row, actual in enumerate(class_ids):
            for column, predicted_id in enumerate(class_ids):
                count = int(counts[row, column])
                if count:
                    mask = (true == actual) & (predicted == predicted_id)
                    matched_paths = [] if image_paths is None else [str(image_paths[index]) for index in np.flatnonzero(mask)]
                    writer.writerow({
                        "true_class_id": int(actual), "true_class": label_for(int(actual)),
                        "predicted_class_id": int(predicted_id), "predicted_class": label_for(int(predicted_id)),
                        "count": count, "true_class_rate": f"{count / totals[row]:.6f}",
                        "true_image_paths": "\n".join(matched_paths),
                    })
    if image_paths is not None:
        with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_path", "true_class_id", "true_class", "predicted_class_id", "predicted_class"])
            writer.writeheader()
            for path_value, actual, predicted_id in zip(image_paths, true, predicted):
                writer.writerow({"image_path": path_value, "true_class_id": int(actual), "true_class": label_for(int(actual)),
                                 "predicted_class_id": int(predicted_id), "predicted_class": label_for(int(predicted_id))})
    errors = sorted(((int(counts[row, column]), label_for(int(actual)), label_for(int(predicted_id)), int(counts[row, column]) / totals[row])
                    for row, actual in enumerate(class_ids) for column, predicted_id in enumerate(class_ids)
                    if counts[row, column] and actual != predicted_id), reverse=True)
    lines = ["# Evaluation metrics", "", "## Summary", "", "| Metric | Value |", "| --- | ---: |"]
    for name, value in metrics.items():
        lines.append(f"| {name} | {value:.6f} |" if isinstance(value, (float, np.floating)) else f"| {name} | {value} |")
    lines.extend(["", "## Confusion matrix", "", f"The full sparse matrix is in [`confusion_matrix.csv`](confusion_matrix.csv). It contains {len(class_ids)} observed classes and {len(errors)} error pairs.", ""])
    if errors:
        lines.extend(["### Most frequent confusions", "", "| True class | Predicted class | Count | Rate within true class |", "| --- | --- | ---: | ---: |"])
        for count, actual, predicted_label, rate in errors[:25]:
            lines.append("| {} | {} | {} | {:.2%} |".format(
                actual.replace("|", "\\|"), predicted_label.replace("|", "\\|"), count, rate,
            ))
    if y_proba is not None:
        proba = np.asarray(y_proba, dtype=float)
        confidence = np.maximum(proba[:, 0], 1 - proba[:, 0]) if proba.shape[1] == 1 else proba.max(axis=1)
        correct, edges, rows = true == predicted, np.linspace(0, 1, 11), []
        for index in range(10):
            low, high = edges[index], edges[index + 1]
            mask = (confidence >= low) & ((confidence <= high) if index == 9 else (confidence < high))
            count = int(mask.sum())
            mean = float(confidence[mask].mean()) if count else float("nan")
            accuracy = float(correct[mask].mean()) if count else float("nan")
            rows.append({"bin_start": low, "bin_end": high, "count": count, "mean_confidence": mean, "accuracy": accuracy})
        with (output / "calibration.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["bin_start", "bin_end", "count", "mean_confidence", "accuracy"])
            writer.writeheader()
            writer.writerows(rows)
        lines.extend(["", "## Calibration", "", "The readable top-label calibration bins are in [`calibration.csv`](calibration.csv).", "", "| Confidence bin | Samples | Mean confidence | Accuracy |", "| --- | ---: | ---: | ---: |"])
        for row in rows:
            lines.append(f"| {row['bin_start']:.1f}–{row['bin_end']:.1f} | {row['count']} | {row['mean_confidence']:.4f} | {row['accuracy']:.4f} |")
    report = output / "README.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
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
        if y_proba.shape[1] not in (1, 2):
            raise ValueError(f"Expected y_proba with shape (*, 1) or (*, 2) for binary, got {y_proba.shape}")
        # One-logit BCE heads provide P(class 1); softmax heads provide both classes.
        positive_proba = y_proba[:, 0] if y_proba.shape[1] == 1 else y_proba[:, 1]
        fpr, tpr, _ = roc_curve(y_true, positive_proba, pos_label=classes[1])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr,
            color=colors[1],
            lw=2,
            label=f"{class_names[1]} (AUC = {roc_auc:.2f})"
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

    # Binary calibration. label_binarize returns one target column for a
    # binary task, even when a softmax model returns two probability columns.
    if num_classes == 2:
        if y_proba.shape[1] not in (1, 2):
            raise ValueError(
                f"Expected y_proba with shape (*, 1) or (*, 2) for binary, got {y_proba.shape}"
            )
        positive_targets = (np.asarray(y_true) == classes[1]).astype(int)
        positive_proba = y_proba[:, 0] if y_proba.shape[1] == 1 else y_proba[:, 1]
        prob_true, prob_pred = calibration_curve(
            positive_targets, positive_proba, n_bins=n_bins
        )
        plt.plot(
            prob_pred, prob_true,
            marker=markers[1 % len(markers)],
            linestyle="-",
            lw=2,
            label=class_names[1],
            color=colors[1]
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
