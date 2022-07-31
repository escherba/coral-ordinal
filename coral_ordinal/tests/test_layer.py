"""
Layer unit tests
"""
from typing import Tuple

import tempfile
import pytest

import numpy as np
from tensorflow.keras import models, layers

try:
    from keras.testing_infra.test_utils import layer_test
except ImportError:
    from keras.testing_utils import layer_test

from coral_ordinal.layer import CoralOrdinal, CornOrdinal
from coral_ordinal.types import IntArray, FloatArray


def _create_test_data() -> Tuple[FloatArray, IntArray]:
    """Fixture that provides data
    Test data from example in
    https://github.com/Raschka-research-group/coral-pytorch/blob/main/coral_pytorch/losses.py
    """
    random_state = np.random.RandomState(10)
    X = random_state.normal(size=(8, 99))
    y = np.array([0, 1, 2, 2, 2, 3, 4, 4])
    return X, y


@pytest.mark.parametrize(
    "klass", [(CornOrdinal), (CoralOrdinal)],
)
def test_corn_layer_builtin(klass: layers.Layer) -> None:
    """Class passes `layer_test`"""
    layer_test(
        klass,
        kwargs={"num_classes": 4},
        input_shape=(10, 5),
    )


def test_corn_layer() -> None:
    """Creation from config works"""
    corn_layer = CornOrdinal(num_classes=4, kernel_initializer="uniform")
    corn_layer_config = corn_layer.get_config()
    corn_layer2 = CornOrdinal(**corn_layer_config)
    assert isinstance(corn_layer2, CornOrdinal)


@pytest.mark.parametrize(
    "klass", [(CornOrdinal), (CoralOrdinal)],
)
def test_serializing_layers(klass: layers.Layer) -> None:
    """Layer serialization works"""
    X, _ = _create_test_data()
    model = models.Sequential()
    model.add(layers.Dense(5, input_dim=X.shape[1]))
    model.add(klass(num_classes=4))
    model.compile(loss="mse")

    preds = model.predict(X)
    with tempfile.TemporaryDirectory() as d:
        models.save_model(model, d)

        model_tmp = models.load_model(d)
        assert isinstance(model_tmp.layers[-1], klass)
    preds_tmp = model_tmp.predict(X)
    np.testing.assert_allclose(preds, preds_tmp)
