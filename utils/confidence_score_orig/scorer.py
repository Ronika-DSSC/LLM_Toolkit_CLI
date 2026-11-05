"""
Combines the individual signals in metrics.py into one 0-1 confidence
score per file, plus a per-metric breakdown for the quality report.

How the final score is computed:
  1. Run every applicable metric (some depend on the source type — e.g.
     cleaning_stability only applies when there's a raw-vs-cleaned pair).
  2. Drop any metric that reported `available=False` (optional dependency
     missing, LLM signal unavailable, etc).
  3. Renormalize the remaining weights so they sum to 1.0, then take the
     weighted mean.

Renormalizing matters: without it, a missing optional dependency would
silently drag every score down toward zero rather than simply being
excluded. 

Weighting rationale: lexical validity and OCR cleanliness carry
the most weight because they directly measure character-level corruption,
which is the dominant failure mode for scanned documents. Completeness
guards against silent truncation. Language coherence is deliberately
lighter, it's excellent at catching total gibberish but nearly blind to
subtle corruption, so weighting it heavily would over-reward text that is
merely language-shaped. Cleaning stability is lightest: it's an indirect
proxy (it measures the source's noisiness, not the output's quality).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from utils.confidence_score.metrics import (
    MetricResult,
    cleaning_stability,
    extraction_completeness,
    language_coherence,
    lexical_validity,
    llm_self_assessment,
    ocr_cleanliness,
)


def _weight(metric_name: str, default: float) -> float:
    env_key = f"CONFIDENCE_WEIGHT_{metric_name.upper()}"
    try:
        return float(os.environ.get(env_key, default))
    except ValueError:
        return default


DEFAULT_WEIGHTS = {
    "lexical_validity": _weight("lexical_validity", 0.30),
    "ocr_cleanliness": _weight("ocr_cleanliness", 0.25),
    "extraction_completeness": _weight("extraction_completeness", 0.25),
    "language_coherence": _weight("language_coherence", 0.15),
    "cleaning_stability": _weight("cleaning_stability", 0.10),
    "llm_self_assessment": _weight("llm_self_assessment", 0.30),
}

USE_LLM_ASSESSMENT = os.environ.get("CONFIDENCE_USE_LLM_ASSESSMENT", "TRUE").lower() == "true"

EXPECTED_LANGUAGE = os.environ.get("CONFIDENCE_EXPECTED_LANGUAGE", "") or None


@dataclass
class ConfidenceResult:
    score: float  # final 0-1 blended confidence
    metrics: list[MetricResult] = field(default_factory=list)

    @property
    def contributing(self) -> list[MetricResult]:
        return [m for m in self.metrics if m.available]

    @property
    def skipped(self) -> list[MetricResult]:
        return [m for m in self.metrics if not m.available]

    def as_dict(self) -> dict[str, Any]:
        """Flat, JSON-serializable form, used in markdown frontmatter and the report."""
        return {
            "confidence_score": self.score,
            "metrics": {
                m.name: {
                    "score": m.score,
                    "raw": m.raw,
                    "available": m.available,
                    "detail": m.detail,
                }
                for m in self.metrics
            },
        }


def score_extraction(
    text: str,
    *,
    raw_text: Optional[str] = None,
    num_pages: Optional[int] = None,
    llm_client: Any = None,
) -> ConfidenceResult:
    """
    Score a single extraction/cleaning result.

    """
    metrics: list[MetricResult] = []

    if not text or not text.strip():
        # Hard override, no output means no confidence.
        metrics.append(MetricResult("extraction_completeness", 0.0, 0, detail="no text extracted"))
        return ConfidenceResult(score=0.0, metrics=metrics)

    metrics.append(lexical_validity(text))
    metrics.append(ocr_cleanliness(text))
    metrics.append(extraction_completeness(text, num_pages=num_pages))
    metrics.append(language_coherence(text, expected_lang=EXPECTED_LANGUAGE))

    if raw_text is not None:
        metrics.append(cleaning_stability(raw_text, text))

    if USE_LLM_ASSESSMENT and llm_client is not None:
        metrics.append(llm_self_assessment(text, llm_client))

    score = _blend(metrics)
    return ConfidenceResult(score=score, metrics=metrics)


def _blend(metrics: list[MetricResult]) -> float:
    """
    Weighted *geometric* mean over available metrics, renormalized.
    """
    usable = [m for m in metrics if m.available and m.name in DEFAULT_WEIGHTS]
    if not usable:
        return 0.0

    total_weight = sum(DEFAULT_WEIGHTS[m.name] for m in usable)
    if total_weight <= 0:
        return 0.0

    epsilon = 0.01
    log_sum = sum(
        DEFAULT_WEIGHTS[m.name] * math.log(max(m.score, epsilon)) for m in usable
    )
    score = math.exp(log_sum / total_weight)
    return round(max(0.0, min(1.0, score)), 3)
