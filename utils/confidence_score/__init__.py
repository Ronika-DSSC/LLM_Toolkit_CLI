"""
Multi-signal confidence scoring for extracted/cleaned text.

"""

from utils.confidence_score.metrics import MetricResult
from utils.confidence_score.scorer import ConfidenceResult, score_extraction

__all__ = ["score_extraction", "ConfidenceResult", "MetricResult"]
