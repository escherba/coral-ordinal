"""Tests activations."""

import numpy as np
from numpy.testing import assert_allclose

import tensorflow as tf

from coral_ordinal import activations


def test_ordinal_probs_to_label() -> None:
    """Conversion from ordinal probability to label"""
    probas = np.array(
        [[0.934, 0.861, 0.323, 0.492, 0.295],
         [0.496, 0.485, 0.267, 0.124, 0.058],
         [0.985, 0.967, 0.920, 0.819, 0.506]]
    )
    labels = activations.cumprobs_to_label(probas).numpy()
    assert_allclose(labels, np.array([2, 0, 5]))


def test_coral_ordinal_softmax() -> None:
    """Test CORAL ordinal logit-to-proba conversion"""
    result = activations.coral_ordinal_softmax(
        tf.constant([[-1, 1], [-2, 2]], dtype=tf.float32))
    expected = np.array(
        [[ 0.7310586 , -0.4621171 ,  0.73105854],
         [ 0.8807971 , -0.7615941 ,  0.880797  ]],
        dtype=np.float32)
    assert_allclose(result, expected, atol=1e-5, rtol=1e-5)


def test_corn_ordinal_softmax() -> None:
    """Test CORN ordinal logit-to-proba conversion"""
    result = activations.corn_ordinal_softmax(
        tf.constant([[-1, 1], [-2, 2]], dtype=tf.float32))
    expected = np.array(
        [[ 0.7310586 , 0.07232951, 0.19661193],
         [ 0.8807971 , 0.01420934, 0.10499357]],
        dtype=np.float32)
    assert_allclose(result, expected, atol=1e-5, rtol=1e-5)
