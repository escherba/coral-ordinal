"""Functions to convert logits to probabilities (CDF), probabilities to laabels, and softmax.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
def coral_cumprobs(logits: tf.Tensor) -> tf.Tensor:
    """Turns logits from CORAL layer into cumulative probabilities."""
    return tf.math.sigmoid(logits)


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
def corn_cumprobs(logits: tf.Tensor, axis: int = 1) -> tf.Tensor:
    """Turns logits from CORN layer into cumulative probabilities."""
    probs = tf.math.sigmoid(logits)
    return tf.math.cumprod(probs, axis=axis)


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
def cumprobs_to_softmax(cumprobs: tf.Tensor) -> tf.Tensor:
    """Turns ordinal probabilities into label probabilities (softmax)."""

    # Number of columns is the number of classes - 1
    num_classes = cumprobs.shape[1] + 1

    # Create a list of tensors.
    # First, get probability predictions out of the cumulative logits.
    # Column 0 is Probability that y > 0, so Pr(y = 0) = 1 - Pr(y > 0)
    # Pr(Y = 0) = 1 - s(logit for column 0)
    probs = [1.0 - cumprobs[:, 0]]

    # For the other columns, the probability is:
    # Pr(y = k) = Pr(y > k) - Pr(y > k - 1)
    if num_classes > 2:
        for val in range(1, num_classes - 1):
            probs.append(cumprobs[:, val - 1] - cumprobs[:, val])

    # Special handling of the maximum label value.
    probs.append(cumprobs[:, num_classes - 2])

    # Combine as columns into a new tensor.
    return tf.concat(tf.transpose(probs), 1)


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
def cumprobs_to_label(cumprobs: tf.Tensor, threshold: float = 0.5) -> tf.Tensor:
    """Converts cumulative probabilities for ordinal data to a class label.

    Converts probabilities of the form

        [Pr(y > 0), Pr(y > 1), ..., Pr(y > K-1)]

    to a predicted label as one of [0, ..., K-1].

    By default, it uses the natural threshold of 0.5 to pick the label.
    Can be changed to be more/less conservative.

    Args:
      cumprobs: tensor with cumulative probabilities from 0..K-1.
      threshold: which threshold to choose for the label prediction.
        Defaults to the natural threshold of 0.5.

    Returns:
      A tensor of one column, with the label (integer).
    """
    assert 0 < threshold < 1, f"threshold must be in (0, 1). Got {threshold}."
    predict_levels = tf.cast(cumprobs > threshold, dtype=tf.int32)
    return tf.reduce_sum(predict_levels, axis=1)


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
def coral_ordinal_softmax(logits: tf.Tensor) -> tf.Tensor:
    """Convert CORAL ordinal logits to label probabilities.

    Args:
        logits: Logit output of `CoralOrdinal()` layer.
    Returns:
        tf.Tensor: probabilities of each class (column) for each sample (row)
    """
    return cumprobs_to_softmax(coral_cumprobs(logits))


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
def corn_ordinal_softmax(logits: tf.Tensor) -> tf.Tensor:
    """Convert CORN ordinal logits to label probabilities.

    Args:
        logits: Logit output of `CornOrdinal()` layer.
    Returns:
        tf.Tensor: probabilities of each class (column) for each sample (row)
    """
    return cumprobs_to_softmax(corn_cumprobs(logits))
