"""
Individual confidence metrics. Each metric is an independent, cheap-to-compute signal about how good an
extraction/cleaning result is. Every one returns a `MetricResult` with a
normalized 0-1 `score` (1.0 = good), the underlying `raw` value it was
derived from, and an `available` flag.

The aggregator in scorer.py drops unavailable metrics and
renormalizes the weights over whatever remains, so the toolkit degrades
gracefully instead of silently scoring everything against a metric that
never actually ran.

"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
import threading
from typing import Any, Optional

# spaCy: real tokenizer + vocabulary for lexical validity
try:  
    import spacy

    try:
        _NLP = spacy.load(
            "en_core_web_sm",
            exclude=[
                "tok2vec", "tagger", "parser", "ner",
                "lemmatizer", "attribute_ruler", "senter", "morphologizer",
            ],
        )
        _SPACY_HAS_VECTORS = _NLP.vocab.vectors.n_keys > 0
        _HAS_SPACY = True
    except Exception:  # model not downloaded
        _NLP = None
        _SPACY_HAS_VECTORS = False
        _HAS_SPACY = False
except ImportError:
    _NLP = None
    _SPACY_HAS_VECTORS = False
    _HAS_SPACY = False

# langdetect: language coherence
try:  
    from langdetect import DetectorFactory, detect_langs

    DetectorFactory.seed = 0  # deterministic results across runs
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False

_SPACY_LOCK = threading.Lock()
_LANGDETECT_LOCK = threading.Lock()

@dataclass
class MetricResult:
    name: str
    score: float  # normalized 0-1, higher is better
    raw: Any  # underlying measurement, for the report/debugging
    available: bool = True
    detail: str = ""

    @staticmethod
    def unavailable(name: str, reason: str) -> "MetricResult":
        return MetricResult(name=name, score=0.0, raw=None, available=False, detail=reason)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_TOKEN_RE = re.compile(r"\S+")

# Characters that legitimately show up in documents. Anything outside this is treated as scanning/encoding garbage.
_LEGIT_CHARS_RE = re.compile(r"[A-Za-z0-9\s\.,;:!\?'\"()\[\]{}<>/\\\-—–_=+*&%$#@°±§¶•|~`^\u00c0-\u017f]")

# "1nv01ce"-style digit/letter substitution inside a word
_MIXED_ALNUM_RE = re.compile(r"\b(?=\w*[A-Za-z])(?=\w*\d)\w{3,}\b")

# Runs of 3+ identical punctuation marks (".....", "-----") — common OCR artifact from dotted leader lines, table borders, and scan edges.
_PUNCT_RUN_RE = re.compile(r"([^\w\s])\1{2,}")

# Isolated single letters that aren't real one-letter words
_ISOLATED_CHAR_RE = re.compile(r"(?<![\w'])[b-hj-z](?![\w'])", re.IGNORECASE)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


# 1. LEXICAL VALIDITY

def lexical_validity(text: str) -> MetricResult:
    """
    Catches character-level OCR corruption, the kind that keeps text
    looking superficially text-shaped while mangling the actual words.

    Uses spaCy's vocabulary when available (checks each token against a
    real lexicon); otherwise falls back to a word-shape heuristic that
    rejects tokens with implausible letter patterns.
    """
    name = "lexical_validity"
    if not text or not text.strip():
        return MetricResult(name, 0.0, 0.0, detail="empty text")

    words = _WORD_RE.findall(text)
    if not words:
        return MetricResult(name, 0.0, 0.0, detail="no alphabetic tokens found")

    if _HAS_SPACY and _NLP is not None:
        # Cap the amount of text sent through spaCy — scoring doesn't need
        # the whole document, and very large docs would be slow.
        with _SPACY_LOCK:
            doc = _NLP(text[:100_000])
        alpha = [t for t in doc if t.is_alpha]
        if not alpha:
            return MetricResult(name, 0.0, 0.0, detail="no alphabetic tokens found")

        if _SPACY_HAS_VECTORS:
            valid = sum(1 for t in alpha if not t.is_oov or len(t.text) <= 2)
            source = "spaCy vocab"
        else:
            # No vectors -> is_oov is unusable (see the load block above).
            # Still a win: spaCy's tokenizer handles contractions,
            # hyphenation and punctuation far better than a regex.
            valid = sum(1 for t in alpha if _looks_like_word(t.text))
            source = "spaCy tokenizer + word-shape heuristic"

        ratio = _safe_ratio(valid, len(alpha))
        return MetricResult(
            name, round(ratio, 4), round(ratio, 4), detail=f"{source}, {len(alpha)} tokens"
        )

    valid = sum(1 for w in words if _looks_like_word(w))
    ratio = _safe_ratio(valid, len(words))
    return MetricResult(
        name,
        round(ratio, 4),
        round(ratio, 4),
        detail=f"heuristic word-shape, {len(words)} tokens (install spaCy for better accuracy)",
    )

def _looks_like_word(word: str) -> bool:
    """
    Plausibility check used when spaCy isn't available.

    Rejects the shapes OCR corruption tends to produce, rather than trying
    to validate against a dictionary.
    """
    w = word.lower().strip("'-")
    if not w:
        return False
    if len(w) == 1:
        return w in ("a", "i")
    if len(w) > 25:  # implausibly long — usually joined/garbled text
        return False
    if not re.search(r"[aeiouy]", w):  # no vowel at all
        return False
    if re.search(r"(.)\1{2,}", w):  # same letter 3+ times ("aaa")
        return False
    if re.search(r"[bcdfghjklmnpqrstvwxz]{5,}", w):  # 5+ consonants in a row
        return False
    return True


# 2. LANGUAGE COHERENCE

def language_coherence(text: str, expected_lang: Optional[str] = None) -> MetricResult:
    """
    Catching whole-output gibberish (a failed OCR pass returning
    noise). 
    """
    name = "language_coherence"
    if not _HAS_LANGDETECT:
        return MetricResult.unavailable(name, "langdetect not installed")
    if not text or len(text.strip()) < 20:
        return MetricResult(name, 0.0, None, detail="text too short to detect reliably")

    try:
        with _LANGDETECT_LOCK:
            candidates = detect_langs(text[:10_000])
    except Exception as e:  # langdetect raises on no-features input
        return MetricResult(name, 0.0, None, detail=f"detection failed: {e}")

    if not candidates:
        return MetricResult(name, 0.0, None, detail="no language detected")

    top = candidates[0]
    if expected_lang:
        match = next((c for c in candidates if c.lang == expected_lang), None)
        prob = match.prob if match else 0.0
        return MetricResult(
            name, round(prob, 4), {"lang": expected_lang, "prob": round(prob, 4)},
            detail=f"expected '{expected_lang}', top was '{top.lang}'",
        )

    return MetricResult(
        name, round(top.prob, 4), {"lang": top.lang, "prob": round(top.prob, 4)},
        detail=f"detected '{top.lang}'",
    )


# 3. OCR NOISE 

def ocr_cleanliness(text: str) -> MetricResult:
    """
    Inverse of an OCR-noise score: 1.0 = clean, 0.0 = very noisy.

    """
    name = "ocr_cleanliness"
    if not text or not text.strip():
        return MetricResult(name, 0.0, None, detail="empty text")

    total_chars = len(text)
    legit_chars = len(_LEGIT_CHARS_RE.findall(text))
    garbage_ratio = 1.0 - _safe_ratio(legit_chars, total_chars)

    words = _WORD_RE.findall(text)
    tokens = _TOKEN_RE.findall(text)
    mixed_alnum = len(_MIXED_ALNUM_RE.findall(text))
    mixed_ratio = _safe_ratio(mixed_alnum, max(len(tokens), 1))

    punct_runs = len(_PUNCT_RUN_RE.findall(text))
    punct_ratio = _safe_ratio(punct_runs, max(len(tokens), 1))

    isolated = len(_ISOLATED_CHAR_RE.findall(text))
    isolated_ratio = _safe_ratio(isolated, max(len(words), 1))

    # Each artifact is scaled before being capped, because these ratios are already alarming at small values 
    penalty = (
        min(garbage_ratio * 8, 1.0) * 0.40
        + min(mixed_ratio * 4, 1.0) * 0.20
        + min(punct_ratio * 5, 1.0) * 0.15
        + min(isolated_ratio * 3, 1.0) * 0.25
    )
    score = max(0.0, 1.0 - penalty)

    raw = {
        "garbage_char_ratio": round(garbage_ratio, 4),
        "mixed_alnum_ratio": round(mixed_ratio, 4),
        "punct_run_ratio": round(punct_ratio, 4),
        "isolated_char_ratio": round(isolated_ratio, 4),
    }
    return MetricResult(name, round(score, 4), raw)


# 4. EXTRACTION COMPLETENESS 

_MIN_CHARS_PER_PAGE = 200


def extraction_completeness(
    text: str, num_pages: Optional[int] = None, source_bytes: Optional[int] = None
) -> MetricResult:
    """
    For paged documents, scores output length against a per page floor.
    Without page info, falls back to a simple absolute length ramp so
    near-empty output is still penalized.
    """
    name = "extraction_completeness"
    stripped = (text or "").strip()
    if not stripped:
        return MetricResult(name, 0.0, 0, detail="no text extracted")

    length = len(stripped)

    if num_pages and num_pages > 0:
        expected = _MIN_CHARS_PER_PAGE * num_pages
        ratio = _safe_ratio(length, expected)
        return MetricResult(
            name,
            round(ratio, 4),
            {"chars": length, "pages": num_pages, "chars_per_page": round(length / num_pages, 1)},
            detail=f"vs ~{_MIN_CHARS_PER_PAGE} chars/page floor",
        )

    # No page count (images, txt, docx): ramp up to a "clearly substantive"
    # length so tiny outputs score low without hard-capping normal ones.
    ratio = min(1.0, math.log1p(length) / math.log1p(_MIN_CHARS_PER_PAGE * 2))
    return MetricResult(
        name, round(ratio, 4), {"chars": length}, detail="length-based (no page count available)"
    )


# CLEANING STABILITY

def cleaning_stability(raw_text: str, cleaned_text: str) -> MetricResult:
    """
    Heavy edits imply a noisier source. Kept as a signal because it's the
    one metric that compares *before vs after* rather than judging the
    final text in isolation.
    """
    name = "cleaning_stability"
    if not cleaned_text or not cleaned_text.strip():
        return MetricResult(name, 0.0, None, detail="empty cleaned text")
    if not raw_text or not raw_text.strip():
        return MetricResult.unavailable(name, "no raw text to compare against")

    raw_len = len(raw_text)
    change_ratio = abs(raw_len - len(cleaned_text)) / max(raw_len, 1)
    score = max(0.0, 1.0 - min(change_ratio, 1.0))
    return MetricResult(name, round(score, 4), round(change_ratio, 4), detail="length delta")


# # LLM SELF-ASSESSMENT
#
# _SELF_ASSESS_PROMPT = (
#     "You are assessing the quality of text extracted from a document, to "
#     "decide whether it needs manual review.\n\n"
#     "Rate the text on a 0-100 scale, where:\n"
#     "  100 = clean, coherent, complete — clearly a faithful extraction\n"
#     "   50 = readable but with noticeable errors, garbling, or gaps\n"
#     "    0 = unusable garbage, or clearly truncated/empty\n\n"
#     "Judge only the extraction quality, NOT whether the underlying "
#     "document's subject matter is interesting or well written.\n\n"
#     "Respond with ONLY a single integer between 0 and 100. No commentary, "
#     "no explanation, no units.\n\n"
#     "---\n{sample}\n---"
# )
#
# _SELF_ASSESS_SAMPLE_CHARS = 3000
#
#
# def llm_self_assessment(text: str, llm_client: Any) -> MetricResult:
#     """
#     Ask the LLM to rate the extraction quality directly.
#     """
#     name = "llm_self_assessment"
#     if not text or not text.strip():
#         return MetricResult(name, 0.0, None, detail="empty text")
#
#     if not hasattr(llm_client, "assess_quality"):
#         return MetricResult.unavailable(name, "llm_client has no assess_quality()")
#
#     sample = text[:_SELF_ASSESS_SAMPLE_CHARS]
#     try:
#         result = llm_client.assess_quality(_SELF_ASSESS_PROMPT.format(sample=sample))
#     except Exception as e:  # noqa: BLE001 — never let a scoring call fail the pipeline
#         return MetricResult.unavailable(name, f"LLM call failed: {e}")
#
#     rating = _parse_rating(result)
#     if rating is None:
#         return MetricResult.unavailable(name, f"could not parse rating from: {result!r:.80}")
#
#     return MetricResult(name, round(rating / 100.0, 4), rating, detail="LLM 0-100 rating")
#
#
# def _parse_rating(response: str) -> Optional[float]:
#     """Pull the first plausible 0-100 integer out of the model's reply."""
#     if response is None:
#         return None
#     match = re.search(r"\b(\d{1,3})\b", str(response))
#     if not match:
#         return None
#     value = float(match.group(1))
#     if value < 0 or value > 100:
#         return None
#     return value
