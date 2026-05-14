"""
Hand-crafted morphology feature vectors for archaic Sinhala Ayurvedic SBD.

These vectors are designed to be **concatenated** with the contextual
embedding produced by a transformer (XLM-RoBERTa) before the final
classification head. The motivation:

* XLM-RoBERTa's SentencePiece tokenizer fragments Sinhala words into
  sub-word pieces that **do not preserve morpheme boundaries**. The model
  must re-learn morphology from scratch from the small available data.
* In Ayurvedic recipe text, sentence ends correlate strongly with a small
  set of verb-final morphemes (Section 6 of THESIS.md). Encoding these as
  explicit dense features short-circuits the morphology-from-scratch
  bottleneck.

The output is a 16-dimensional binary feature vector, designed to be
small enough to avoid dominating the 768-dim RoBERTa embedding but rich
enough to encode the linguistic signal.

This module is the *novel architectural contribution* of Phase-2: a
**morphology-injected token classification head** for low-resource SBD.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from config import CANONICAL_ENDINGS, normalize_sinhala
from labeling_functions import (
    CLINICAL_VERB_LEMMAS,
    EXTENDED_SUFFIXES,
    NEGATION_MARKERS,
)


# Index 0..12 — one bit per canonical suffix (13 features).
# Index 13   — any extended suffix (broader than canonical).
# Index 14   — clinical-verb lemma.
# Index 15   — negation marker.
# (16 total)
MORPH_FEATURE_DIM = 16


def morph_vector(word: str) -> np.ndarray:
    """Compute the 16-dim morphology vector for a single word."""
    w = normalize_sinhala(word)
    v = np.zeros(MORPH_FEATURE_DIM, dtype=np.float32)
    for k, suf in enumerate(CANONICAL_ENDINGS):
        if w.endswith(suf):
            v[k] = 1.0
    if any(w.endswith(s) for s in EXTENDED_SUFFIXES):
        v[13] = 1.0
    if w in CLINICAL_VERB_LEMMAS:
        v[14] = 1.0
    if w in NEGATION_MARKERS:
        v[15] = 1.0
    return v


def morph_matrix(words: Sequence[str]) -> np.ndarray:
    """Stack morphology vectors for a sequence of words → (L, 16)."""
    return np.stack([morph_vector(w) for w in words], axis=0)
