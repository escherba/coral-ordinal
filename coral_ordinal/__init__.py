"""
Tensorflow Keras implementation of CORAL ordinal regression
"""
from .version import __version__

from .layer import CoralOrdinal, CornOrdinal
from .loss import CoralOrdinalCrossEntropy, CornOrdinalCrossEntropy
from .metrics import MeanAbsoluteErrorLabels
from .activations import (
    coral_ordinal_softmax,
    corn_ordinal_softmax,
    cumprobs_to_label,
    corn_cumprobs,
    coral_cumprobs,
)

# if somebody does "from somepackage import *", this is what they will
# be able to access:
__all__ = [
    "CoralOrdinal",
    "CornOrdinal",
    "MeanAbsoluteErrorLabels",
    "CoralOrdinalCrossEntropy",
    "CornOrdinalCrossEntropy",
    "coral_ordinal_softmax",
    "corn_ordinal_softmax",
    "cumprobs_to_label",
    "coral_cumprobs",
    "corn_cumprobs",
]
