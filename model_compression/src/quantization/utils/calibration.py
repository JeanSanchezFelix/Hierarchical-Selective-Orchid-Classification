import torch
import tensorflow as tf 
import numpy as np  # NumPy for data manipulation
from typing import Iterator, Generator, List

def representative_data_gen(
    dataloader: torch.utils.data.DataLoader,
    num_samples: int = 100,
) -> Iterator[List[tf.Tensor]]:
    """
    Generate representative samples for TFLite calibration from a PyTorch DataLoader.

    Converts images from CHW PyTorch format to NHWC TensorFlow tensors for quantization calibration.

    Args:
        dataloader: DataLoader yielding (images, labels), where images are CHW tensors.
        num_samples: Maximum number of samples to yield.

    Yields:
        A list containing a single TensorFlow Tensor of shape (1, H, W, C) and dtype float32.

    Raises:
        StopIteration: When the requested number of samples have been generated.
    """
    yielded = 0
    for images, _ in dataloader:
        for img in images:
            if yielded >= num_samples:
                return
            # Convert CHW to HWC numpy array
            arr = img.detach().cpu().numpy()            # (C,H,W)
            arr = np.transpose(arr, (1,2,0)).astype(np.float32)  # (H,W,C)
            # Create TF tensor with batch dimension
            tf_tensor = tf.expand_dims(tf.convert_to_tensor(arr), 0)  # (1,H,W,C)
            yielded += 1
            yield [tf_tensor]

### For SavedModel -> TfLite (not tested) ###

# def representative_data_generator(
#     dataset: tf.data.Dataset,
#     num_samples: int = 100,
# ) -> Generator[List[np.ndarray], None, None]:
#     """
#     Generate representative data for TFLite static quantization calibration.

#     Args:
#         dataset (tf.data.Dataset): A dataset yielding (input, label) tuples.
#         num_samples (int): Number of samples to use for calibration. Defaults to 100.

#     Yields:
#         List[np.ndarray]: A list containing a single batch of input data as NumPy arrays.
#     """
#     # Iterate over the dataset up to num_samples
#     for _, (input_data, _) in enumerate(dataset.take(num_samples)):
#         # Convert Tensor to NumPy array if necessary
#         data = input_data.numpy() if hasattr(input_data, 'numpy') else np.array(input_data)
#         # Yield list-of-arrays as expected by TFLiteConverter
#         yield [data]
