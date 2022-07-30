"""
Custom types module
"""
from typing import Tuple, List, Callable, Union, TypeVar, Any  # pylint: disable=unused-import

import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor

import numpy as np  # pylint: disable=unused-import


T = TypeVar("T")

# generic factory type (either instance class or instance builder)
Factory = Union[T, Callable[..., T]]

# Using a Union trick to avoid values below being interpreted as string
# literals.  When NumPy v1.21 stops being so common, one should get rid of both
# Union and strings below.
AnyArray = Union[
    'np.ndarray[Any, Any]',
    'np.ndarray[Any, Any]']
IntArray = Union[
    'np.ndarray[Any, np.dtype[np.integer[Any]]]',
    'np.ndarray[Any, np.dtype[np.integer[Any]]]']
FloatArray = Union[
    'np.ndarray[Any, np.dtype[np.floating[Any]]]',
    'np.ndarray[Any, np.dtype[np.floating[Any]]]']
BoolArray = Union[
    'np.ndarray[Any, np.dtype[np.bool_]]',
    'np.ndarray[Any, np.dtype[np.bool_]]']


NoneType = type(None)

KerasBlock = Callable[[KerasTensor], KerasTensor]

Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

TensorLike = Union[
    List[Union[Number, List[Number]]],
    Tuple[Number, ...],
    Number,
    IntArray,
    FloatArray,
    BoolArray,
    tf.Tensor,
    tf.SparseTensor,
    tf.Variable,
    KerasTensor,
]

Initializer = Union[None, dict, str, tf.keras.initializers.Initializer]
Regularizer = Union[None, dict, str, tf.keras.regularizers.Regularizer]
Constraint = Union[None, dict, str, tf.keras.constraints.Constraint]
Activation = Union[None, str, Callable[[tf.Tensor], tf.Tensor]]
