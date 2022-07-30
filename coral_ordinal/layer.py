"""
Coral-ordinal layers
"""
# pylint: disable=attribute-defined-outside-init
# pylint: disable=too-few-public-methods
from typing import Optional, Any, Dict
import warnings
import tensorflow as tf
from tensorflow.keras import regularizers

from .types import Regularizer, Activation


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
class CoralOrdinal(tf.keras.layers.Layer):
    """Implements the CORAL ordinal layer for ordinal regression."""

    # We skip input_dim/input_shape here and put in the build() method as
    # recommended in the tutorial, in case the user doesn't know the input
    # dimensions when defining the model.
    def __init__(
        self,
        num_classes: int,
        activation: Optional[str] = None,
        kernel_regularizer: Regularizer = None,
        bias_regularizer: Regularizer = None,
        **kwargs: Any,
    ) -> None:
        """Ordinal output layer, which produces ordinal logits by default.

        Args:
          num_classes: how many ranks (aka labels or values) are in the ordinal variable.
          activation: (Optional) Activation function to use. The default of None produces
            ordinal logits, but passing "ordinal_softmax" will cause the layer to output
            a probability prediction for each label.
          kernel_regularizer: regularizer for kernel of Coral Dense layer.
          bias_regularizer: regularizer for bias of Coral Dense layer.
          **kwargs: keyword arguments passed to Layer().
        """

        # Via Dense Layer code:
        # https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/layers/core.py#L1128
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)

        # Pass any additional keyword arguments to Layer() (i.e. name, dtype)
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.activation = tf.keras.activations.get(activation)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

    # Following
    # https://www.tensorflow.org/guide/keras/custom_layers_and_models#best_practice_deferring_weight_creation_until_the_shape_of_the_inputs_is_known
    def build(self, input_shape: tf.TensorShape) -> None:
        """Set up weights from input shape"""
        # Single fully-connected neuron - this is the latent variable.
        num_units = 1

        # I believe glorot_uniform (aka Xavier uniform) is pytorch's default initializer, per
        # https://pytorch.org/docs/master/generated/torch.nn.Linear.html
        # and https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
        self.kernel = self.add_weight(
            shape=(input_shape[-1], num_units),
            # Need a unique name if there are multiple coral_ordinal layers.
            name=self.name + "_latent",
            initializer="glorot_uniform",
            regularizer=self.kernel_regularizer,
            # Not sure if this is necessary:
            dtype=tf.float32,
            trainable=True,
        )

        # num_classes - 1 bias terms, defaulting to 0.
        self.bias = self.add_weight(
            shape=(self.num_classes - 1,),
            # Need a unique name if there are multiple coral_ordinal layers.
            name=self.name + "_bias",
            regularizer=self.bias_regularizer,
            initializer="zeros",
            # Not sure if this is necessary:
            dtype=tf.float32,
            trainable=True,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Layer forward pass"""

        kernelized_inputs = tf.matmul(inputs, self.kernel)

        logits = kernelized_inputs + self.bias

        if self.activation is None:
            outputs = logits
        else:
            # Not yet tested:
            outputs = self.activation(logits)

        return outputs

    # This allows for serialization.
    # https://www.tensorflow.org/guide/keras/custom_layers_and_models#you_can_optionally_enable_serialization_on_your_layers
    def get_config(self) -> Dict[str, Any]:
        """Return config for serialization"""
        config = {
            "num_classes": self.num_classes,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
class CornOrdinal(tf.keras.layers.Dense):
    """Implements the CORN layer for ordinal regression."""

    # We skip input_dim/input_shape here and put in the build() method as
    # recommended in the tutorial, in case the user doesn't know the input
    # dimensions when defining the model.
    def __init__(
            self,
            num_classes: int,
            activation: Activation = None,
            **kwargs: Any) -> None:
        """Ordinal output layer, which produces ordinal logits by default.

        Args:
          num_classes: how many ranks (aka labels or values) are in the ordinal variable.
          activation: (Optional) Activation function to use. The default of None produces
            ordinal logits, but passing "ordinal_softmax" will cause the layer to output
            a probability prediction for each label.
        """
        if "units" in kwargs:
            warnings.warn("Use 'num_classes' instead of 'units'. Dropping ...")
            kwargs.pop("units")

        super().__init__(units=num_classes - 1, activation=activation, **kwargs)
        if activation is not None:
            raise NotImplementedError(
                f"CornOrdinal() must return logits. Got {activation}."
            )
        self.num_classes = num_classes
        self.activation = activation

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serializing"""
        config = {
            "num_classes": self.num_classes,
            "activation": self.activation
        }
        base_config = super().get_config()
        base_config.pop("units")
        return {**base_config, **config}
