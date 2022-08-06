"""Module for testing """
# pylint: disable=invalid-name
from typing import Tuple

import pytest
import numpy as np
import tensorflow as tf
from keras import models, layers, losses

from coral_ordinal.layer import CornOrdinal, CoralOrdinal
from coral_ordinal.loss import CoralOrdinalCrossEntropy, CornOrdinalCrossEntropy
from coral_ordinal.types import IntArray, FloatArray
from coral_ordinal.utils import encode_ordinal_labels_numpy


def _create_test_data() -> Tuple[FloatArray, IntArray, IntArray]:
    """Fixture that creates test data
    The data is from example in
    https://github.com/Raschka-research-group/coral-pytorch/blob/main/coral_pytorch/losses.py
    """
    random_state = np.random.RandomState(10)
    X = random_state.normal(size=(8, 99))
    y = np.array([0, 1, 2, 2, 2, 3, 4, 4])
    sample_weights = np.array([0, 1, 1, 1, 1, 1, 1, 1])
    return X, y, sample_weights


def test_CoralOrdinalCrossentropy() -> None:
    """basic dense correctness test"""
    loss = CoralOrdinalCrossEntropy(sparse=False)
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1, 1.]]))
    expect = tf.constant(1.6265233)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseCoralOrdinalCrossentropy() -> None:
    """basic sparse correctness test"""
    loss = CoralOrdinalCrossEntropy(sparse=True)
    val = loss(tf.constant([[2.]]), tf.constant([[-1, 1.]]))
    expect = tf.constant(1.6265233)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_CornOrdinalCrossentropy() -> None:
    """basic dense correctness test"""
    loss = CornOrdinalCrossEntropy(sparse=False)
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1, 1.]]))
    expect = tf.constant(0.54217446)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseCornOrdinalCrossentropy() -> None:
    """basic sparse correctness test"""
    loss = CornOrdinalCrossEntropy(sparse=True)
    val = loss(tf.constant([[2.]]), tf.constant([[-1, 1.]]))
    expect = tf.constant(0.54217446)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "klass",
    [CoralOrdinalCrossEntropy, CornOrdinalCrossEntropy],
)
def test_SparseOrderInvariant(klass: losses.Loss) -> None:
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
    [CoralOrdinalCrossEntropy, CornOrdinalCrossEntropy],
)
def test_SparseInequality(klass: losses.Loss) -> None:
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


def test_corn_loss() -> None:
    """Corn loss correctness test"""
    X, y, _ = _create_test_data()
    corn_loss = CornOrdinalCrossEntropy()
    num_classes = len(np.unique(y))
    tf.random.set_seed(1)
    corn_net = CornOrdinal(num_classes=num_classes, input_dim=X.shape[1])
    logits = corn_net(X)
    assert logits.shape == (8, num_classes - 1)

    loss_val = corn_loss(y_true=y, y_pred=logits)
    # see https://github.com/Raschka-research-group/coral-pytorch/blob/main/coral_pytorch/losses.py
    # for approximately same value for pytorch immplementation.
    # Divide by sample size = 8 here since TF defaults to sum over batch size, not sum.
    assert loss_val.numpy() == pytest.approx(3.54 / 8.0, 0.01)


@pytest.mark.parametrize(
    "reduction,expected_len",
    [("auto", 1), ("none", 8), ("sum", 1), ("sum_over_batch_size", 1)],
)
def test_coral_loss_reduction(reduction: str, expected_len: int) -> None:
    """Coral loss reduction works"""
    X, y, _ = _create_test_data()
    coral_loss = CoralOrdinalCrossEntropy(reduction=reduction)
    num_classes = len(np.unique(y))

    tf.random.set_seed(1)
    coral_net = CoralOrdinal(num_classes=num_classes, input_dim=X.shape[1])
    logits = coral_net(X)
    loss_val = coral_loss(y_true=y, y_pred=logits)
    if expected_len == 1:
        assert loss_val.numpy() > 0
    else:
        assert loss_val.shape[0] == expected_len


@pytest.mark.parametrize(
    "reduction,expected_len",
    [("auto", 1), ("none", 8), ("sum", 1), ("sum_over_batch_size", 1)],
)
def test_corn_loss_reduction(reduction: str, expected_len: int) -> None:
    """Corn loss reduction works"""
    X, y, _ = _create_test_data()
    corn_loss = CornOrdinalCrossEntropy(reduction=reduction)
    num_classes = len(np.unique(y))

    tf.random.set_seed(1)
    corn_net = CornOrdinal(num_classes=num_classes, input_dim=X.shape[1])
    logits = corn_net(X)
    loss_val = corn_loss(y_true=y, y_pred=logits)
    if expected_len == 1:
        assert loss_val.numpy() > 0
    else:
        assert loss_val.shape[0] == expected_len


@pytest.mark.parametrize(
    "y_lt",
    [1, 4, 5],
)
def test_sample_weights_loss(y_lt: int) -> None:
    """loss calculation also works when not all labels are present."""
    X, y, sample_weights = _create_test_data()
    corn_loss = CornOrdinalCrossEntropy(reduction="none")
    num_classes = len(np.unique(y))

    tf.random.set_seed(1)
    corn_net = CornOrdinal(num_classes=num_classes, input_dim=X.shape[1])
    logits = corn_net(X)

    y_lt_mask = y < y_lt
    loss_val = corn_loss(y_true=y[y_lt_mask], y_pred=logits[y_lt_mask]).numpy()
    loss_val_weighted = corn_loss(
        y_true=y[y_lt_mask],
        y_pred=logits[y_lt_mask],
        sample_weight=sample_weights[y_lt_mask],
    ).numpy()

    np.testing.assert_allclose(loss_val * sample_weights[y_lt_mask], loss_val_weighted)


def test_sample_weight_in_fit() -> None:
    """sample weights work during `fit()`"""
    X, y, _ = _create_test_data()
    weight = np.zeros_like(y)
    model = models.Sequential()
    model.add(layers.Dense(5, input_dim=X.shape[1]))
    model.add(CornOrdinal(num_classes=len(np.unique(y))))
    model.compile(loss=CornOrdinalCrossEntropy())

    history = model.fit(X, y, sample_weight=weight, epochs=2)
    np.testing.assert_allclose(np.array(history.history["loss"]), np.array([0.0, 0.0]))


def test_class_weight_in_fit() -> None:
    """class weights work during `fit()`"""
    X, y, _ = _create_test_data()
    model = models.Sequential()
    model.add(layers.Dense(5, input_dim=X.shape[1]))
    model.add(CornOrdinal(num_classes=len(np.unique(y))))
    model.compile(loss=CornOrdinalCrossEntropy())

    history = model.fit(
        X, y, epochs=2, class_weight={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    )
    np.testing.assert_allclose(np.array(history.history["loss"]), np.array([0.0, 0.0]))
