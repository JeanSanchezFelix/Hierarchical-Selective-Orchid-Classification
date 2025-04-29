import numpy as np
from tqdm import tqdm

from model_compression.src.utils.metrics import calculate_metrics, plot_metric_bar, plot_confusion_matrix, plot_roc_auc_curve, plot_radar_chart

try:
    import tensorflow as tf
except ImportError:
    tf = None

def evaluate_tflite_model(
    tflite_model_path: str,
    test_dataset: tf.data.Dataset,
    input_type: tf.dtypes.DType = tf.uint8,
) -> float:
    """
    Evaluate a TFLite model's accuracy on a test dataset.

    Args:
        tflite_model_path (str): Path to the .tflite model file.
        test_dataset (tf.data.Dataset): Dataset yielding (input, label) tuples.
        input_type (tf.dtypes.DType): Expected input dtype for the model. Defaults to tf.uint8.

    Returns:
        float: Classification accuracy (0.0 - 1.0).
    """
    # Load the TFLite model into an interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    # Allocate necessary tensors
    interpreter.allocate_tensors()
    # Obtain input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct_predictions = 0
    total_samples = 0

    # Iterate through the test dataset
    for input_data, label in test_dataset:
        # Prepare input array with correct dtype
        array = input_data.numpy() if hasattr(input_data, 'numpy') else np.array(input_data)
        # Cast to expected numpy dtype
        array = array.astype(input_type.as_numpy_dtype)
        # Add batch dimension if missing
        if array.ndim == len(input_details[0]['shape']) - 1:
            array = np.expand_dims(array, axis=0)
        # Set tensor and invoke interpreter
        interpreter.set_tensor(input_details[0]['index'], array)
        interpreter.invoke()
        # Retrieve output and compute predicted label
        output = interpreter.get_tensor(output_details[0]['index'])
        pred = int(np.argmax(output, axis=-1))

        # Compare prediction to ground-truth
        true_label = int(label.numpy()) if hasattr(label, 'numpy') else int(label)
        if pred == true_label:
            correct_predictions += 1
        total_samples += 1

    # Return accuracy
    return correct_predictions / total_samples if total_samples else 0.0