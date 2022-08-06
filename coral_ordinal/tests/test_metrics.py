"""Tests for coral metric function."""
from typing import Tuple, List, Union

import pytest
import numpy as np
import tensorflow as tf

from coral_ordinal.metrics import MeanAbsoluteErrorLabels
from coral_ordinal.utils import encode_ordinal_labels_numpy


def test_config() -> None:
    """basic configuration test"""
    mael_obj = MeanAbsoluteErrorLabels()
    assert mael_obj.name == "mean_absolute_error_labels"
    assert mael_obj.dtype == tf.float32


def get_data() -> Tuple[List[List[Union[int, List[int]]]], List[List[List[float]]]]:
    """fixture to get data arrays used in testing"""
    actuals: List[List[Union[int, List[int]]]] = [
        [[7], [2], [1]],
        [0, 0, 0],
        [0, 0, 0],
    ]
    preds: List[List[List[float]]] = [
        [
            [10.9, 6.3, 4.7, 3.4, 2.5, 1.8, 0.8, -0.4, -2.2],
            [5.9, 1.3, -0.2, -1.4, -2.3, -3.1, -4.1, -5.3, -7.1],
            [2.9, -1.6, -3.2, -4.5, -5.4, -6.1, -7.1, -8.4, -10.2],
        ],
        [
            [10.9, 6.3, 4.7, 3.4, 2.5, 1.8, 0.8, -0.4, -2.2],
            [5.9, 1.3, -0.2, -1.4, -2.3, -3.1, -4.1, -5.3, -7.1],
            [2.9, -1.6, -3.2, -4.5, -5.4, -6.1, -7.1, -8.4, -10.2],
        ],
        [
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9],
        ],
    ]
    return actuals, preds


def test_dense_ordinal_mae_mismatch() -> None:
    """basic dense correctness test"""
    loss = MeanAbsoluteErrorLabels(sparse=False)
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1, 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_sparse_ordinal_mae_mismtatch() -> None:
    """basic sparse correctness test"""
    loss = MeanAbsoluteErrorLabels(sparse=True)
    val = loss(tf.constant([[2.]]), tf.constant([[-1, 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_dense_ordinal_mae_match() -> None:
    """basic dense correctness test"""
    loss = MeanAbsoluteErrorLabels(sparse=False)
    val = loss(tf.constant([[1., 1.]]), tf.constant([[1, 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_sparse_ordinal_mae_match() -> None:
    """basic sparse correctness test"""
    loss = MeanAbsoluteErrorLabels(sparse=True)
    val = loss(tf.constant([[2.]]), tf.constant([[1, 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "klass",
    [MeanAbsoluteErrorLabels],
)
def test_sparse_order_invariance(klass: type) -> None:
    """test order invariance (equal after shuffling)"""
    for _ in range(10):
        num_classes = np.random.randint(2, 8)
        loss = klass(sparse=True)
        y_true = np.random.randint(0, num_classes, 20)
        y_pred1 = encode_ordinal_labels_numpy(
            y_true, num_classes=num_classes)
        observed1 = loss(y_true, y_pred1)
        np.random.shuffle(y_true)
        y_pred2 = encode_ordinal_labels_numpy(
            y_true, num_classes=num_classes)
        observed2 = loss(y_true, y_pred2)
        tf.debugging.assert_near(observed1, observed2)


@pytest.mark.parametrize(
    "klass",
    [MeanAbsoluteErrorLabels],
)
def test_sparse_inequality(klass: type) -> None:
    """test expected inequality (equal or worse after shuffling)"""
    for _ in range(10):
        num_classes = np.random.randint(2, 8)
        loss = klass(sparse=True)
        y_true = np.random.randint(0, num_classes, 20)
        y_true_orig = y_true.copy()
        y_pred1 = encode_ordinal_labels_numpy(
            y_true_orig, num_classes=num_classes)
        observed1 = loss(y_true_orig, y_pred1)
        np.random.shuffle(y_true)
        y_pred2 = encode_ordinal_labels_numpy(
            y_true, num_classes=num_classes)
        observed2 = loss(y_true_orig, y_pred2)
        tf.debugging.assert_less_equal(observed1, observed2)


def test_mae_labels_score() -> None:
    """MAE labels score correctness"""
    actuals, preds = get_data()

    mael_obj1 = MeanAbsoluteErrorLabels()
    mael_obj1.update_state(
        tf.constant(actuals[0], dtype=tf.int32), tf.constant(preds[0], dtype=tf.float32)
    )
    # [7, 2, 1] - [7, 2, 1] = 0
    np.testing.assert_allclose(0.0, mael_obj1.result())

    mael_obj2 = MeanAbsoluteErrorLabels()
    mael_obj2.update_state(
        tf.constant(actuals[1], dtype=tf.int32), tf.constant(preds[1], dtype=tf.float32)
    )
    # [7, 2, 1] - [0, 0, 0] = (7 + 2 + 1) / 3 = 3.3333333333
    np.testing.assert_allclose(3.3333333333, mael_obj2.result())

    mael_obj3 = MeanAbsoluteErrorLabels()
    mael_obj3.update_state(
        tf.constant(actuals[2], dtype=tf.int32), tf.constant(preds[2], dtype=tf.float32)
    )
    # [0, 0, 0] - [0, 0, 0] = 0
    np.testing.assert_allclose(0.0, mael_obj3.result())


def test_mae_labels_running_score() -> None:
    """MAE labels running score correctness"""
    mael_obj = MeanAbsoluteErrorLabels()
    actuals, preds = get_data()
    for actual, pred in zip(actuals, preds):
        mael_obj.update_state(
            tf.constant(actual, dtype=tf.int32), tf.constant(pred, dtype=tf.float32)
        )
    # (0 + 3.3333333333 + 0) / 3 = 1.1111111111
    np.testing.assert_allclose(1.1111111111, mael_obj.result())

    mael_obj.reset_state()
    np.testing.assert_allclose(0.0, mael_obj.result())
