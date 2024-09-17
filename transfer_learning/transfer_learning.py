import os
import csv
import argparse
import configparser
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as tfk                          # needed due to incompatability with tensorflow_hub version

from sklearn.metrics import confusion_matrix, classification_report, recall_score, f1_score, ConfusionMatrixDisplay
from tensorflow.keras.optimizers import Adam, SGD


# Function to load arguments from a config file (.csv)
def load_args_from_file(file_path):
    args_from_file = {}
    
    # If the file is .csv, read it as key-value pairs
    if file_path.endswith('.csv'):
        with open(file_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, value in row.items():
                    args_from_file[key] = value.strip() if value else None  # Strip spaces or set to None if empty               
    else:
        raise ValueError("Unsupported file format. Please provide a .csv file.")

    return args_from_file


# Argument Parsing
parser = argparse.ArgumentParser(description='Train multiple models with custom configurations.')
parser.add_argument('--config_file', type=str, help='Path to configuration file (CSV)')
parser.add_argument('--model_name', type=str, help='Name of the pre-trained model to use (mobilenet, efficientnet, etc.)')
parser.add_argument('--num_models', type=int, help='Number of models to train')
parser.add_argument('--num_epochs', type=int, help='Number of epochs to train each model')
parser.add_argument('--dataset_dir', type=str, help='Directory where the dataset is located')
parser.add_argument('--batch_size', type=int, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer')
parser.add_argument('--optimizer', type=str, choices=["adam", "sgd"], help='Optimizer to use: adam or sgd')
parser.add_argument('--save_dir', type=str, help='Directory to save the models')

# Parse command-line arguments
args = parser.parse_args()

# If a configuration file is provided, load arguments from the file
config_args = {}
if args.config_file:
    config_args = load_args_from_file(args.config_file)

# Set default or override config file arguments with command-line arguments
MODEL_NAME = args.model_name or config_args.get('model_name', 'mobilenet')
NUM_MODELS = args.num_models or int(config_args.get('num_models', 1))
NUM_EPOCHS = args.num_epochs or int(config_args.get('num_epochs', 5))
dataset_dir = args.dataset_dir or config_args.get('dataset_dir')
batch_size = args.batch_size or int(config_args.get('batch_size', 32))
learning_rate = args.learning_rate or float(config_args.get('learning_rate', 0.001))
optimizer = args.optimizer or config_args.get('optimizer', 'adam')
save_dir = args.save_dir or config_args.get('save_dir', 'saved_models')

# Argument Validation
if NUM_MODELS <= 0:
    raise ValueError("The number of models (--num_models) must be a positive integer.")
if NUM_EPOCHS <= 0:
    raise ValueError("The number of epochs (--num_epochs) must be a positive integer.")
if batch_size <= 0:
    raise ValueError("The number of epochs (--batch_size) must be a positive integer.")
if learning_rate <= 0:
    raise ValueError("The number of epochs (--learning_rate) must be a positive integer.")
if not os.path.exists(dataset_dir):
    raise ValueError(f"The specified dataset directory {dataset_dir} does not exist.")

# Print the argument values for verification
print(f"MODEL_NAME: {MODEL_NAME}")
print(f"NUM_MODELS: {NUM_MODELS}")
print(f"NUM_EPOCHS: {NUM_EPOCHS}")
print(f"dataset_dir: {dataset_dir}")
print(f"batch_size: {batch_size}")
print(f"learning_rate: {learning_rate}")
print(f"optimizer: {optimizer}")
print(f"save_dir: {save_dir}")

# Pre-trained model URLs from TensorFlow Hub
model_urls = {
    "mobilenet": "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
    "efficientnet": "https://tfhub.dev/google/efficientnet/b0/feature-vector/1",
    "resnet50": "https://tfhub.dev/tensorflow/resnet_50/feature_vector/1",
    "inceptionv3": "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4",
}

# Ensure the model name is valid
if MODEL_NAME not in model_urls:
    raise ValueError(f"Unsupported model name '{MODEL_NAME}'. Supported models: {list(model_urls.keys())}")

# Create directory for saving models
os.makedirs(save_dir, exist_ok=True)

# Get the model URL based on the selected model name
feature_extractor_model = model_urls[MODEL_NAME]

# Define optimizer
if args.optimizer == "adam":
    optimizer = tfk.optimizers.Adam(learning_rate=learning_rate)
elif args.optimizer == "sgd":
    optimizer = tfk.optimizers.SGD(learning_rate=learning_rate),

# Load the dataset from the specified directory
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=batch_size
)

class_names = np.array(train_ds.class_names)    # class labels for Monkeypox dataset
num_classes = len(class_names)

# TODO: Might need to make normalization optional depending on the model (eg. mobilenet needs normaliztion but others might not)
normalization_layer = tf.keras.layers.Rescaling(1./255)           # normalize pixel values of images from [0, 255] to [0, 1] by dividing
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) # normalize training split where x—images, y—labels.
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))     # normalize validation split where x—images, y—labels.

AUTOTUNE = tf.data.AUTOTUNE
# prefetch data to improve performance by overlapping data preprocessing and model execution and cache the dataset in memory
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Define plotting and evaluation functions
def save_confusion_matrix(true_labels, predicted_labels, class_names, save_path):
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

def save_loss_curve(history, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='orange')
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def save_evaluation_metrics(true_labels, predicted_labels, history, cm, save_path):
    accuracy = history['val_accuracy'][-1]
    sensitivity = recall_score(true_labels, predicted_labels, average='macro')
    specificity = np.mean(np.diag(cm) / (np.diag(cm) + np.sum(cm, axis=0) - np.diag(cm)))
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    metrics = {
        "Accuracy": accuracy,
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "F1-Score": f1
    }

    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values(), color=['darkturquoise', 'sandybrown', 'hotpink', 'limegreen'])
    plt.title("Model Evaluation Metrics")
    plt.ylim([0, 1])
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylabel("Score")
    plt.savefig(save_path)
    plt.close()
    return metrics

def save_classification_report(true_labels, predicted_labels, class_names, save_path):
    class_report = classification_report(true_labels, predicted_labels, target_names=class_names, digits=4)
    with open(save_path, "w") as f:
        f.write(class_report)

# List to store accuracy results for comparison
model_performance = []

for i in range(NUM_MODELS):
    print(f"Training model {i + 1}/{NUM_MODELS}")

    # Create subdirectory for this model
    model_subdir = os.path.join(save_dir, f"model_{i + 1}")
    os.makedirs(model_subdir, exist_ok=True)

    # Load the pre-trained model from TensorFlow Hub (fresh model for each iteration)
    feature_extractor_layer = hub.KerasLayer(
        feature_extractor_model,
        input_shape=(224, 224, 3),
        trainable=False)  # Freeze the base model for transfer learning

    # Define the model
    model = tfk.Sequential([
        feature_extractor_layer,
        tfk.layers.Dense(num_classes, activation='softmax')  # Use softmax for multi-class classification
    ])

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    # Train the model
    history = model.fit(train_ds, validation_data=val_ds, epochs=NUM_EPOCHS)

    # Save the model
    model_path = os.path.join(model_subdir, f"model_{i + 1}.h5")
    model.save(model_path)

    # Save training history
    history_path = os.path.join(model_subdir, f"history_{i + 1}.npy")
    np.save(history_path, history.history)

    # Predictions and true labels
    val_predictions = model.predict(val_ds)
    val_predicted_ids = np.argmax(val_predictions, axis=-1)
    true_labels = np.concatenate([y for x, y in val_ds], axis=0)

    # Save confusion matrix
    confusion_matrix_path = os.path.join(model_subdir, "confusion_matrix.png")
    save_confusion_matrix(true_labels, val_predicted_ids, class_names, confusion_matrix_path)

    # Plot and save loss curve
    loss_curve_path = os.path.join(model_subdir, "loss_curve.png")
    save_loss_curve(history.history, loss_curve_path)

    # Calculate and plot metrics
    cm = confusion_matrix(true_labels, val_predicted_ids)
    bar_chart_path = os.path.join(model_subdir, "evaluation_metrics.png")
    save_evaluation_metrics(true_labels, val_predicted_ids, history.history, cm, bar_chart_path)

    # Save classification report
    classification_report_path = os.path.join(model_subdir, "classification_report.txt")
    save_classification_report(true_labels, val_predicted_ids, class_names, classification_report_path)

    # Record the final validation accuracy for comparison
    final_val_acc = history.history['val_accuracy'][-1]
    model_performance.append((model_path, final_val_acc))

    print(f"Model {i + 1} saved to {model_path} with validation accuracy: {final_val_acc:.4f}")

# After the loop, print out the results for comparison
model_performance.sort(key=lambda x: x[1], reverse=True)
print("\nModels ranked by validation accuracy:")
for i, (model_path, accuracy) in enumerate(model_performance):
    print(f"Model {i + 1}: {model_path}, Validation Accuracy: {accuracy:.4f}")
