"""
Utility functions
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes

from .types import IntArray, FloatArray


def encode_ordinal_labels_numpy(
        array: IntArray,
        num_classes: int,
        skip_last: bool = True,
        dtype: type = np.float32) -> FloatArray:
    """Encoder ordinal data to one-hot type

    Example:

        >>> labels = np.arange(3)
        >>> encode_ordinal_labels_numpy(labels, num_classes=3, skip_last=True)
        array([[0., 0.],
               [1., 0.],
               [1., 1.]], dtype=float32)
        >>> encode_ordinal_labels_numpy(labels, num_classes=3, skip_last=False)
        array([[0., 0., 0.],
               [1., 0., 0.],
               [1., 1., 0.]], dtype=float32)
    """
    compare_to = np.arange(num_classes)
    if skip_last:
        compare_to = compare_to[:-1]
    mask = array[:, None] > compare_to
    return mask.astype(dtype)


def encode_ordinal_labels(
        labels: tf.Tensor,
        num_classes: int,
        dtype: dtypes.DType = tf.float32) -> tf.Tensor:
    """Convert ordinal label to one-hot representation

    Args:
        labels (tf.Tensor): a tensor of ordinal labels (starting with zero)
        num_classes (int): assumed number of classes
        dtype (dtypes.DType): result data type

    Returns:
        tf.Tensor: a tensor of levels (one-hot-encoded labels)

    Example:

        >>> encode_ordinal_labels([0, 1, 2], num_classes=3)
        <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
        array([[0., 0.],
               [1., 0.],
               [1., 1.]], dtype=float32)>

    Calling this is equivalent to:

        levels = [1] * label + [0] * (num_classes - 1 - label)

    """
    # This function uses tf.sequence_mask(), which is vectorized, and avoids
    # map_fn() call.
    return tf.sequence_mask(labels, maxlen=num_classes - 1, dtype=dtype)
