"""
Ordinal loss functions
"""
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
from typing import Optional, Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.keras.losses import Reduction
from keras import losses

from .activations import corn_cumprobs
from .utils import encode_ordinal_labels
from .types import FloatArray


def _coral_ordinal_loss_no_reduction(
        logits: tf.Tensor,
        levels: tf.Tensor,
        importance_weights: Optional[tf.Tensor] = None) -> tf.Tensor:
    """Compute ordinal loss without reduction."""
    levels = tf.cast(levels, dtype=logits.dtype)
    loss_values = (
        tf.math.log_sigmoid(logits) * levels
        + (tf.math.log_sigmoid(logits) - logits) * (1.0 - levels)
    )
    if importance_weights is not None:
        importance_weights = tf.cast(importance_weights, dtype=loss_values.dtype)
        loss_values = tf.multiply(loss_values, importance_weights)
    return -tf.reduce_sum(loss_values, axis=1)


def _reduce_losses(
        values: tf.Tensor,
        reduction: Reduction) -> tf.Tensor:
    """Reduces loss values to specified reduction."""
    if reduction == Reduction.NONE:
        return values

    if reduction in [
        Reduction.AUTO,
        Reduction.SUM_OVER_BATCH_SIZE,
    ]:
        return tf.reduce_mean(values)

    if reduction == Reduction.SUM:
        return tf.reduce_sum(values)

    raise ValueError(f"'{reduction}' is not a valid reduction.")


# The outer function is a constructor to create a loss function using a certain number of classes.
@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
class CoralOrdinalCrossEntropy(losses.Loss):
    """Computes ordinal cross entropy based on ordinal predictions and outcomes."""

    num_classes: Optional[int]
    sparse: bool
    importance_weights: Optional[FloatArray]
    from_type: str

    def __init__(
            self,
            num_classes: Optional[int] = None,
            sparse: bool = True,
            importance_weights: Optional[FloatArray] = None,
            from_type: str = "ordinal_logits",
            name: str = "ordinal_crossentropy",
            **kwargs: Any) -> None:
        """Cross-entropy loss designed for ordinal outcomes.

        Args:
          num_classes: number of ranks (aka labels or values) in the ordinal variable.
            This is optional; can be inferred from size of y_pred at runtime.
          importance_weights: class weights for each binary classification task.
          from_type: one of "ordinal_logits" (default), "logits", or "probs".
            Ordinal logits are the output of a CoralOrdinal() layer with no activation.
            (Not yet implemented) Logits are the output of a dense layer with no activation.
            (Not yet implemented) Probs are the probability outputs of a softmax or ordinal_softmax
            layer.
          name: name of layer
          **kwargs: keyword arguments passed to Loss().
        """
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.sparse = sparse
        self.importance_weights = importance_weights
        self.from_type = from_type

    # Following https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss
    def call(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            sample_weight: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Forward pass"""

        from_type = self.from_type
        y_pred = tf.convert_to_tensor(y_pred)

        if self.num_classes is None:
            self.num_classes = int(y_pred.get_shape().as_list()[1]) + 1

        if self.sparse:
            # Convert each true label to a vector of ordinal level indicators.
            # This also ensures that tf_levels is the same type as y_pred (presumably a float).
            y_true = encode_ordinal_labels(
                tf.squeeze(y_true), self.num_classes, dtype=y_pred.dtype)

        if from_type == "ordinal_logits":
            loss_values = _coral_ordinal_loss_no_reduction(
                y_pred, y_true, self.importance_weights
            )
        elif from_type == "probs":
            raise NotImplementedError("not yet implemented")
        elif from_type == "logits":
            raise NotImplementedError("not yet implemented")
        else:
            raise ValueError(f"Unknown from_type value {from_type}")

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, loss_values.dtype)
            loss_values = tf.multiply(loss_values, sample_weight)

        return _reduce_losses(loss_values, self.reduction)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serializing"""
        config = {
            "num_classes": self.num_classes,
            "sparse": self.sparse,
            "importance_weights": self.importance_weights,
            "from_type": self.from_type,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
class CornOrdinalCrossEntropy(losses.Loss):
    """Implements CORN ordinal loss function for logits.

    Computes the CORN loss described in https://arxiv.org/abs/2111.08851
    """

    num_classes: Optional[int]
    sparse: bool
    importance_weights: Optional[FloatArray]

    def __init__(
            self,
            num_classes: Optional[int] = None,
            sparse: bool = True,
            importance_weights: Optional[FloatArray] = None,
            **kwargs: Any) -> None:
        """Initializes class."""
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.sparse = sparse
        self.importance_weights = importance_weights

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serializing"""
        config = {
            "num_classes": self.num_classes,
            "sparse": self.sparse,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def call(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            sample_weight: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Forward pass

        Args:
          y_true: true labels (0..N-1)
          y_pred: predicted logits (from CornLayer())
          sample_weight: optional; provide sample weights for each sample.

        Returns:
          loss: tf.Tensor, that contains the loss value.
        """
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        y_true = tf.cast(y_true, y_pred.dtype)

        if self.num_classes is None:
            self.num_classes = int(y_pred.get_shape().as_list()[1]) + 1

        if self.sparse:
            tmp = tf.squeeze(y_true)
            sets = [(tmp > i) for i in range(self.num_classes - 1)]
        else:
            tmp = tf.cast(y_true, tf.bool)
            sets = [tmp[:, i] for i in range(self.num_classes - 1)]

        n_examples = tf.shape(y_pred)[0]
        loss_values = tf.zeros(n_examples)
        set_mask = tf.cast(tf.ones(n_examples), tf.bool)

        importance_weights = self.importance_weights
        if importance_weights is None:
            importance_weights = np.ones(self.num_classes - 1)

        assert len(importance_weights) == len(sets)
        for task_index, (label_gt_i, weight) in enumerate(zip(sets, importance_weights)):

            pred_task = tf.gather(y_pred, task_index, axis=1)
            losses_task = tf.where(
                set_mask,
                tf.where(
                    label_gt_i,
                    tf.math.log_sigmoid(pred_task),  # if label > i
                    tf.math.log_sigmoid(pred_task) - pred_task,  # if label <= i
                ),
                0.0,  # don't add to loss if label is <= i - 1
            )
            loss_values -= (losses_task * weight)
            set_mask = label_gt_i
        loss_values /= self.num_classes

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, loss_values.dtype)
            loss_values = tf.multiply(loss_values, sample_weight)

        return _reduce_losses(loss_values, self.reduction)


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
class OrdinalEarthMoversDistance(tf.keras.losses.Loss):
    """"Squared Earth Movers' Distance Loss

    Note: See [2, Eq. 14] and [3, Eq. 10]. Assumptions from [3] interpreted for
    histograms: Compared histograms must (a) have equal mass, and (b) use the same
    bins (umk-wk, umk+wk) for any k.

    Original implementation:
    https://github.com/ldgarcia/canopy-cover-tree-count-estimation/blob/main/dlc/losses/emd.py

    References:

    [1] Avi-Aharon, M., Arbelle, A., & Raviv, T. R. (2020).
        DeepHist: Differentiable Joint and Color Histogram Layers for
        Image-to-Image Translation.
        arXiv preprint arXiv:2005.03995. URL: https://arxiv.org/abs/2005.03995

    [2] Rubner, Y., Tomasi, C., & Guibas, L. J. (2000).
        The earth mover's distance as a metric for image retrieval.
        International journal of computer vision, 40(2), 99-121.

    [3] Hou, L., Yu, C. P., & Samaras, D. (2016).
        Squared earth mover's distance-based loss for training deep neural networks.
        arXiv preprint arXiv:1611.05916. URL: https://arxiv.org/abs/1611.05916
    """

    num_classes: Optional[int]
    sparse: bool
    importance_weights: Optional[FloatArray]
    from_type: str

    def __init__(
            self,
            num_classes: Optional[int] = None,
            sparse: bool = True,
            importance_weights: Optional[FloatArray] = None,
            from_type: str = "ordinal_logits",
            name: str = "ordinal_earth_movers_distance",
            **kwargs: Any) -> None:
        """Squared Earth Movers' Distance Loss
        Args:
          num_classes: number of ranks (aka labels or values) in the ordinal variable.
            This is optional; can be inferred from size of y_pred at runtime.
          importance_weights: class weights for each binary classification task.
          from_type: one of "ordinal_logits" (default), "logits", or "probs".
            Ordinal logits are the output of a CoralOrdinal() layer with no activation.
            (Not yet implemented) Logits are the output of a dense layer with no activation.
            (Not yet implemented) Probs are the probability outputs of a softmax or ordinal_softmax
            layer.
          name: name of layer
          **kwargs: keyword arguments passed to Loss().
        """
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.sparse = sparse
        self.importance_weights = importance_weights
        self.from_type = from_type

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serializing"""
        config = {
            "num_classes": self.num_classes,
            "sparse": self.sparse,
            "importance_weights": self.importance_weights,
            "from_type": self.from_type,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def call(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            sample_weight: Optional[tf.Tensor] = None) -> tf.Tensor:
        """forward pass"""
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        if self.num_classes is None:
            self.num_classes = int(y_pred.get_shape().as_list()[1]) + 1

        importance_weights = self.importance_weights

        if self.num_classes is None:
            self.num_classes = int(y_pred.get_shape().as_list()[1]) + 1

        if self.sparse:
            y_true = encode_ordinal_labels(y_true, num_classes=self.num_classes)

        y_pred = corn_cumprobs(y_pred, axis=-1)
        loss_values = tf.math.squared_difference(y_true, y_pred)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, loss_values.dtype)
            loss_values = tf.multiply(loss_values, sample_weight)

        if importance_weights is not None:
            importance_weights = tf.cast(importance_weights, dtype=loss_values.dtype)
            loss_values = tf.multiply(loss_values, importance_weights)

        return _reduce_losses(loss_values, self.reduction)
