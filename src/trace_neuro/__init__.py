"""TRACE: contrastive learning for multi-trial time-series neural data."""

__version__ = "0.1.0"

from trace_neuro.models import TimeSeriesMLP, TimeSeriesProjectionHead
from trace_neuro.pairs import ContrastiveTrialPairGenerator
from trace_neuro.evaluate import knn_accuracy, ari_score