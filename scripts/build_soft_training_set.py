"""
Build the Phase-2 soft-labelled training set.

What this script does and *why*
-------------------------------
Phase 1's weak-supervision (`notebooks/phase2/weak_supervision_generator.py`)
suffered from two coupled flaws:

1.  Every training sequence was *exactly one* sentence, so STOP only ever
    appeared at the final token.  Any model — CRF or RoBERTa — could
    trivially achieve perfect F1 by learning the rule "STOP = end of
    sequence."  This is what produced the 100% F1 in `evaluate.py`.

2.  Labels came from a single deterministic suffix rule that was *also*
    injected into the CRF as the `is_common_ending` feature.  Training is
    therefore circular (`docs/FULL_AUDIT_REPORT.md` §2.2).

This script repairs both flaws:

*  We **concatenate** 3–8 random consecutive lines from
   ``data/cleaned_corpus.txt`` to form each training sequence, so STOP
   appears at *interior* positions of the sequence — the real SBD task.

*  We apply six independent labelling functions
   (``src/labeling_functions.py``) and combine their votes through a
   Snorkel-style generative label model
   (``src/label_model.py``) to produce *soft* probabilistic labels.

The output is two TSV files plus a JSON summary:

    data/train_soft.tsv          word \t hard_label \t p_stop  (per token)
    data/train_soft_meta.json    fitted LF accuracies + corpus stats
    data/train_hard.tsv          word \t hard_label   (compat with crfsuite)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import List, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np  # noqa: E402

from config import DATA_DIR, normalize_sinhala  # noqa: E402
from labeling_functions import (  # noqa: E402
    ALL_LF_NAMES,
    apply_all_lfs,
    ensure_endword_statistics,
)
from label_model import fit_label_model, predict_proba, summary  # noqa: E402


CORPUS_PATH = os.path.join(DATA_DIR, "cleaned_corpus.txt")
OUT_SOFT = os.path.join(DATA_DIR, "train_soft.tsv")
OUT_HARD = os.path.join(DATA_DIR, "train_hard.tsv")
OUT_META = os.path.join(DATA_DIR, "train_soft_meta.json")


def load_corpus_lines(path: str) -> List[List[str]]:
    """Read the 1-sentence-per-line corpus and return tokenised lines."""
    lines: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            tokens = [normalize_sinhala(t) for t in raw.strip().split() if t.strip()]
            if tokens:
                lines.append(tokens)
    return lines


def build_concatenated_sequences(
    lines: List[List[str]],
    min_sents: int = 3,
    max_sents: int = 8,
    n_sequences: int | None = None,
    seed: int = 13,
) -> List[Tuple[List[str], List[int]]]:
    """Concatenate consecutive lines into multi-sentence training sequences.

    Returns a list of (tokens, line_end_indices) tuples, where
    `line_end_indices` are the global token positions that were last token
    of an *original* corpus line — i.e. ground-truth STOP positions.
    """
    rng = random.Random(seed)
    if n_sequences is None:
        # Aim for similar total tokens as the original training set.
        n_sequences = max(1, len(lines) // ((min_sents + max_sents) // 2))

    seqs: List[Tuple[List[str], List[int]]] = []
    n_lines = len(lines)
    for _ in range(n_sequences):
        k = rng.randint(min_sents, max_sents)
        start = rng.randint(0, max(0, n_lines - k))
        chunk = lines[start : start + k]
        tokens: List[str] = []
        end_idx: List[int] = []
        for sent in chunk:
            tokens.extend(sent)
            end_idx.append(len(tokens) - 1)
        seqs.append((tokens, end_idx))
    return seqs


def main():
    p = argparse.ArgumentParser(description="Build soft-labelled training set.")
    p.add_argument("--min-sents", type=int, default=3, help="Min sentences/sequence.")
    p.add_argument("--max-sents", type=int, default=8, help="Max sentences/sequence.")
    p.add_argument("--n-sequences", type=int, default=12000,
                   help="Number of concatenated sequences to emit.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="P(STOP) threshold for hard-label TSV.")
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()

    print(f"Loading corpus from {CORPUS_PATH}…")
    lines = load_corpus_lines(CORPUS_PATH)
    print(f"  {len(lines):,} lines, "
          f"{sum(len(l) for l in lines):,} tokens.")

    print("Computing corpus end-word statistics (cached)…")
    p_end_stats = ensure_endword_statistics()
    print(f"  vocab size: {len(p_end_stats['p_end']):,}, "
          f"corpus prior P(end) = {p_end_stats['mean_p_end']:.4f}")

    print(f"Concatenating into {args.n_sequences:,} multi-sentence sequences "
          f"(k ∈ [{args.min_sents},{args.max_sents}])…")
    seqs = build_concatenated_sequences(
        lines,
        min_sents=args.min_sents,
        max_sents=args.max_sents,
        n_sequences=args.n_sequences,
        seed=args.seed,
    )

    print("Applying labelling functions to all tokens…")
    all_votes_rows: List[List[int]] = []
    seq_boundaries: List[int] = []  # cumulative offsets to recover sequences later
    seq_tokens_flat: List[str] = []
    seq_line_end_global: List[set] = []
    running = 0
    for tokens, end_idx in seqs:
        votes = apply_all_lfs(tokens, end_idx, p_end_stats)
        all_votes_rows.extend(votes)
        seq_tokens_flat.extend(tokens)
        seq_line_end_global.append({running + e for e in end_idx})
        running += len(tokens)
        seq_boundaries.append(running)

    votes_arr = np.array(all_votes_rows, dtype=np.int64)
    print(f"  vote matrix shape: {votes_arr.shape} (tokens, LFs)")

    print("Fitting label model…")
    fit = fit_label_model(votes_arr, ALL_LF_NAMES, anchor="line_end")
    print(summary(fit))

    proba = predict_proba(votes_arr, fit)
    hard = (proba >= args.threshold).astype(int)

    print(f"Marginal STOP rate (soft): {proba.mean():.4f}")
    print(f"Marginal STOP rate (hard ≥ {args.threshold}): {hard.mean():.4f}")
    print(f"Ground-truth STOP rate (line-end anchor): "
          f"{(votes_arr[:, 0] == 1).mean():.4f}")

    # ------------------------------------------------------------------
    # Emit TSVs and metadata
    # ------------------------------------------------------------------
    print(f"Writing soft labels  → {OUT_SOFT}")
    print(f"Writing hard labels  → {OUT_HARD}")
    soft_f = open(OUT_SOFT, "w", encoding="utf-8")
    hard_f = open(OUT_HARD, "w", encoding="utf-8")
    cursor = 0
    for seq_idx, end in enumerate(seq_boundaries):
        for j in range(cursor, end):
            word = seq_tokens_flat[j]
            hl = "STOP" if hard[j] == 1 else "O"
            p = proba[j]
            soft_f.write(f"{word}\t{hl}\t{p:.6f}\n")
            hard_f.write(f"{word}\t{hl}\n")
        soft_f.write("\n")
        hard_f.write("\n")
        cursor = end
    soft_f.close()
    hard_f.close()

    print(f"Writing meta → {OUT_META}")
    meta = {
        "n_sequences": len(seqs),
        "n_tokens": int(votes_arr.shape[0]),
        "lf_names": fit.lf_names,
        "lf_accuracies": fit.accuracies,
        "lf_weights": fit.weights,
        "lf_coverage": fit.coverage,
        "lf_polarity": fit.polarity,
        "anchor_lf": fit.anchor,
        "threshold": args.threshold,
        "seed": args.seed,
        "min_sents": args.min_sents,
        "max_sents": args.max_sents,
        "marginal_p_stop_soft": float(proba.mean()),
        "marginal_p_stop_hard": float(hard.mean()),
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
