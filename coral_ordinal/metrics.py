"""
Coral-ordinal metrics
"""
from typing import Any, Optional, Dict

import tensorflow as tf
from keras import backend as K
from keras import metrics

from . import activations


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
class MeanAbsoluteErrorLabels(metrics.Metric):
    """Computes mean absolute error for ordinal labels."""

    sparse: bool

    def __init__(
            self,
            sparse: bool = True,
            corn_logits: bool = False,
            threshold: float = 0.5,
            name: str = "mean_absolute_error_labels",
            **kwargs: Any) -> None:
        """Create the state variables

        Creates a `MeanAbsoluteErrorLabels` instance.

        Args:
          corn_logits: if True, inteprets y_pred as CORN logits; otherwise (default)
            as CORAL logits.
          threshold: which threshold should be used to determine the label from
            the cumulative probabilities. Defaults to 0.5.
          name: name of metric.
          **kwargs: keyword arguments passed to parent Metric().
        """
        super().__init__(name=name, **kwargs)
        self.sparse = sparse
        self._corn_logits = corn_logits
        self._threshold = threshold
        self.maes = self.add_weight(name="maes", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            sample_weight: Optional[tf.Tensor] = None) -> None:
        """Update the state variables given y_true and y_pred

        Computes mean absolute error for ordinal labels.

        Args:
          y_true: Labels (int).
          y_pred: Cumulative logits from CoralOrdinal layer.
          sample_weight (optional): sample weights to weight absolute error.
        """

        # Predict the label as in Cao et al. - using cumulative probabilities.
        if self._corn_logits:
            cumprobs = activations.corn_cumprobs(y_pred)
        else:
            cumprobs = activations.coral_cumprobs(y_pred)

        # Threshold cumulative probabilities at predefined cutoff (user set).
        label_pred = tf.cast(
            activations.cumprobs_to_label(cumprobs, threshold=self._threshold),
            dtype=tf.float32,
        )

        if not self.sparse:
            # Sum across columns to estimate how many cumulative thresholds are
            # passed.
            y_true = tf.reduce_sum(y_true, axis=1)

        y_true = tf.cast(y_true, label_pred.dtype)

        # remove all dimensions of size 1 (e.g., from [[1], [2]], to [1, 2])
        y_true = tf.squeeze(y_true)
        label_pred = tf.squeeze(label_pred)
        label_abs_err = tf.abs(y_true - label_pred)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, y_true.dtype)
            sample_weight = tf.broadcast_to(sample_weight, label_abs_err.shape)
            label_abs_err = tf.multiply(label_abs_err, sample_weight)

        self.maes.assign_add(tf.reduce_mean(label_abs_err))
        self.count.assign_add(tf.constant(1.0))

    def result(self) -> tf.Tensor:
        """Return the scalar metric result"""
        return tf.math.divide_no_nan(self.maes, self.count)

    def reset_state(self) -> None:
        """Clear the state at the start of each epoch."""
        K.batch_set_value([(v, 0) for v in self.variables])

    def get_config(self) -> Dict[str, Any]:
        """Returns the serializable config of the metric."""
        config = {
            "sparse": self.sparse,
            "threshold": self._threshold,
            "corn_logits": self._corn_logits,
        }
        base_config = super().get_config()
        return {**base_config, **config}


# # WIP
# def MeanAbsoluteErrorLabels_v2(y_true, y_pred):
#   # There will be num_classes - 1 cumulative logits as columns of the tensor.
#   num_classes = y_pred.shape[1] + 1
#
#   probs = logits_to_probs(y_pred, num_classes)
#
# # RootMeanSquaredErrorLabels
