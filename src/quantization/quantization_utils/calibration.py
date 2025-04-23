import torch
import tensorflow as tf 
from typing import Iterator

def representative_data_gen(dataloader: torch.utils.data.DataLoader) -> Iterator[list[tf.Tensor]]:
    """
    Yield batches from a DataLoader for TFLite calibration.

    Args:
        dataloader (DataLoader): PyTorch DataLoader for the calibration set.

    Yields:
        Iterator of single-element lists containing input tensors as NumPy arrays.
    """
    for batch in dataloader:
        images, _ = batch
        # Convert to NumPy and wrap in list per TFLite API
        yield [images.numpy()]