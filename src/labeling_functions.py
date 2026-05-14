"""
Multi-rule weak-supervision labelling functions for Sinhala Ayurvedic SBD.

This module is the Phase-2 replacement for the single-rule weak supervisor in
notebooks/phase2/weak_supervision_generator.py.

Design goal
-----------
Six independent labelling functions (LFs) emit a noisy label in {O, STOP,
ABSTAIN} for every token in a sentence. A generative label model
(`src/label_model.py`) then combines them into a probabilistic soft label.

By having multiple, partially redundant signals — none of which is the same
function used as a CRF/RoBERTa feature — we break the circular training
dynamic identified in `docs/FULL_AUDIT_REPORT.md` §2.2.

Label semantics
---------------
Every LF returns one of:
    LABEL_O      = 0   (token is NOT a sentence boundary)
    LABEL_STOP   = 1   (token IS a sentence boundary)
    LABEL_ABSTAIN = -1 (LF has no opinion)

LFs intentionally abstain on the cases they cannot judge. The label model
combines non-abstain votes weighted by LF accuracy.
"""
from __future__ import annotations

import json
import math
import os
from typing import List, Sequence

from config import (
    CANONICAL_ENDINGS,
    WEAK_SUPERVISION_EXTRA_ENDINGS,
    DATA_DIR,
    normalize_sinhala,
)

LABEL_O = 0
LABEL_STOP = 1
LABEL_ABSTAIN = -1


# -----------------------------------------------------------------------------
# Auxiliary corpus statistics used by the data-driven LFs
# -----------------------------------------------------------------------------
def build_endword_statistics(corpus_path: str) -> dict:
    """Compute per-word P(word is line-final) statistics from a 1-sentence-per-line corpus.

    The training corpus (`data/cleaned_corpus.txt`) was pre-segmented so each
    line is exactly one sentence. We use this property as a high-quality
    distant-supervision signal: for every word, count how often it terminates
    a line vs. appears mid-line.

    Returns
    -------
    dict with keys:
        end_count[word]    -> times word appeared as the last token of a line
        total_count[word]  -> total occurrences of word
        p_end[word]        -> end_count / total_count  (smoothed)
        mean_p_end         -> corpus-wide prior
    """
    end_count: dict = {}
    total_count: dict = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
            if not words:
                continue
            for i, w in enumerate(words):
                w = normalize_sinhala(w)
                total_count[w] = total_count.get(w, 0) + 1
                if i == len(words) - 1:
                    end_count[w] = end_count.get(w, 0) + 1

    p_end = {}
    for w, total in total_count.items():
        ends = end_count.get(w, 0)
        # Laplace smoothing: pretend we saw one of each
        p_end[w] = (ends + 1.0) / (total + 2.0)

    if total_count:
        mean_p_end = sum(end_count.values()) / max(sum(total_count.values()), 1)
    else:
        mean_p_end = 0.1

    return {
        "end_count": end_count,
        "total_count": total_count,
        "p_end": p_end,
        "mean_p_end": mean_p_end,
    }


def save_endword_statistics(stats: dict, out_path: str) -> None:
    """Persist a slim version of the stats (only p_end + mean) for runtime use."""
    slim = {"p_end": stats["p_end"], "mean_p_end": stats["mean_p_end"]}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False)


def load_endword_statistics(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Clinical imperative / declarative verbs frequently terminating Ayurvedic
# recipe steps (compiled from manual inspection of cleaned_corpus.txt).
# -----------------------------------------------------------------------------
CLINICAL_VERB_LEMMAS = {
    "මැනවි", "ගනු", "තබනු", "කරනු", "ආලේපය", "පානය",
    "පොවනු", "ගල්වනු", "තවරනු", "තවරන්න", "තබන්න",
    "පොවන්න", "ආලේප", "තැවරිය", "ගිලින්න",
}

NEGATION_MARKERS = {
    "නෑ", "නැත", "නොමැත", "නොකරන්න", "එපා", "නොකර", "නොකරයි",
}

EXTENDED_SUFFIXES = list(set(CANONICAL_ENDINGS + WEAK_SUPERVISION_EXTRA_ENDINGS + [
    "ය", "කරන්න", "දෙන්න",
]))


# -----------------------------------------------------------------------------
# Labelling functions
# -----------------------------------------------------------------------------
def lf_line_end(words: Sequence[str], i: int, *, is_line_end: bool) -> int:
    """LF_lineend — corpus structure: the last token of a line IS a sentence boundary.

    This is the highest-precision LF because the corpus was pre-segmented
    one sentence per line. Note: it only votes STOP, never O; abstaining when
    not at end-of-line keeps it from drowning the other signals.
    """
    return LABEL_STOP if is_line_end else LABEL_ABSTAIN


def lf_canonical_suffix(words: Sequence[str], i: int, **_) -> int:
    """LF_canon — token ends in a canonical Sinhala sentence-final suffix.

    Votes STOP on hit, abstains on miss (does not vote O, because many
    non-boundary words also fail to end in any canonical suffix).
    """
    w = words[i]
    if any(w.endswith(s) for s in CANONICAL_ENDINGS):
        return LABEL_STOP
    return LABEL_ABSTAIN


def lf_extended_suffix(words: Sequence[str], i: int, **_) -> int:
    """LF_ext — broader suffix set including weak-supervision-only endings."""
    w = words[i]
    if any(w.endswith(s) for s in EXTENDED_SUFFIXES):
        return LABEL_STOP
    return LABEL_ABSTAIN


def lf_clinical_verb(words: Sequence[str], i: int, **_) -> int:
    """LF_clinverb — token is a known Ayurvedic clinical verb lemma."""
    if words[i] in CLINICAL_VERB_LEMMAS:
        return LABEL_STOP
    return LABEL_ABSTAIN


def lf_negation(words: Sequence[str], i: int, **_) -> int:
    """LF_neg — negation markers at the end of an instruction.

    Negations in Ayurvedic instructions ("නොකරන්න") are typically clause-final;
    a strong but rarer signal.
    """
    if words[i] in NEGATION_MARKERS:
        return LABEL_STOP
    return LABEL_ABSTAIN


def lf_corpus_pend(
    words: Sequence[str], i: int, *,
    p_end_stats: dict,
    tau_high: float = 0.30,
    tau_low: float = 0.02,
) -> int:
    """LF_pend — purely data-driven: P(word terminates a line) from corpus stats.

    Thresholds are calibrated against the corpus marginal end-rate
    (≈0.09 for the Ayurvedic recipe corpus): a word that ends a sentence
    >3× more often than chance fires STOP; a word that ends one >4× less
    often than chance fires O; otherwise abstain.

    This LF is the only one that can vote O, giving the label model a
    counterweight to suffix-rule false positives.
    """
    w = words[i]
    p_end_map = p_end_stats.get("p_end") if "p_end" in p_end_stats else p_end_stats
    p = p_end_map.get(w)
    if p is None:
        return LABEL_ABSTAIN
    if p >= tau_high:
        return LABEL_STOP
    if p <= tau_low:
        return LABEL_O
    return LABEL_ABSTAIN


ALL_LF_NAMES = [
    "line_end",
    "canon_suffix",
    "ext_suffix",
    "clinical_verb",
    "negation",
    "corpus_pend",
]


def apply_all_lfs(
    words: Sequence[str],
    line_end_indices: Sequence[int],
    p_end_stats: dict,
) -> List[List[int]]:
    """Apply every LF to every token in `words`.

    Parameters
    ----------
    words : list of tokens already Unicode-normalised
    line_end_indices : positions in `words` that are KNOWN sentence ends
        (typically the index of the last token of each original line, when
        sentences have been concatenated to build a long sequence)
    p_end_stats : dict from `load_endword_statistics`

    Returns
    -------
    matrix : list of length len(words), each entry is a list of 6 LF votes
             in the order defined by ALL_LF_NAMES.
    """
    end_set = set(line_end_indices)
    out: List[List[int]] = []
    for i, _ in enumerate(words):
        is_le = i in end_set
        row = [
            lf_line_end(words, i, is_line_end=is_le),
            lf_canonical_suffix(words, i),
            lf_extended_suffix(words, i),
            lf_clinical_verb(words, i),
            lf_negation(words, i),
            lf_corpus_pend(words, i, p_end_stats=p_end_stats),
        ]
        out.append(row)
    return out


# -----------------------------------------------------------------------------
# Default p_end stats path
# -----------------------------------------------------------------------------
DEFAULT_PEND_PATH = os.path.join(DATA_DIR, "endword_statistics.json")


def ensure_endword_statistics(corpus_path: str = None, out_path: str = None) -> dict:
    """Idempotently build and cache the corpus end-word statistics."""
    if corpus_path is None:
        corpus_path = os.path.join(DATA_DIR, "cleaned_corpus.txt")
    if out_path is None:
        out_path = DEFAULT_PEND_PATH
    if os.path.exists(out_path):
        return load_endword_statistics(out_path)
    stats = build_endword_statistics(corpus_path)
    save_endword_statistics(stats, out_path)
    return load_endword_statistics(out_path)
