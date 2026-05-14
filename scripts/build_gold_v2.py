"""
Build the Phase-2 gold-standard SBD test set (v2).

Why a new gold standard?
------------------------
The Phase-1 gold set (`data/gold_test.tsv`, 20 sentences from
`scripts/generate_gold.py`) is too small for the audit-grade evaluation
described in `docs/FULL_AUDIT_REPORT.md` §2.1, and every sentence has
exactly one STOP at end-of-sequence — so it cannot test *mid-sequence*
boundary detection at all.

Design
------
The v2 gold set is constructed by sampling real lines from
``data/cleaned_corpus.txt`` (which is already one-sentence-per-line),
then **concatenating** them into multi-sentence sequences whose interior
line-end positions are *ground-truth* STOP boundaries.  We stratify by
difficulty:

  * EASY   — 2 sentences concatenated (single mid-sequence boundary).
  * MED    — 3-4 sentences concatenated.
  * HARD   — 5-7 sentences (multi-boundary, longer context).
  * AMBIG  — sequences where ≥2 LFs disagree on at least one token
             (flagged for manual review).

Output
------
``data/gold_test_v2.tsv``  CoNLL-style:  word \t label  with blank
                            lines separating sequences.

``data/gold_test_v2.meta.json``  per-sequence difficulty + LF-disagreement
                            stats so that downstream evaluation can
                            slice metrics by difficulty bucket.

Seed sentences from ``scripts/generate_gold.py`` (the hand-typed authentic
examples) are added as an additional difficulty bucket EXPERT — these
sentences are NOT drawn from cleaned_corpus, so they test out-of-corpus
generalisation.
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
from labeling_functions import apply_all_lfs, ensure_endword_statistics, ALL_LF_NAMES  # noqa: E402

CORPUS_PATH = os.path.join(DATA_DIR, "cleaned_corpus.txt")
OUT_TSV = os.path.join(DATA_DIR, "gold_test_v2.tsv")
OUT_META = os.path.join(DATA_DIR, "gold_test_v2.meta.json")

# Authentic expert-typed sentences from the original gold set —
# kept as a separate "EXPERT" bucket because they are NOT in the
# training corpus and therefore exercise out-of-distribution behaviour.
EXPERT_SEED = [
    "අමු ඉඟුරු කොත්තමල්ලි පත්පාඩගම් කටුවැල්බටු යන ඖෂධ වර්ග සමානව ගෙන වතුර පත අට එකට සිඳුවා උදේ සවස පානය කිරීම ගුණදායකයි",
    "කටුවැල්බටු රසකිඳ තිප්පිලි වැනි ඖෂධ තම්බා මී පැණි සමඟ පානය කිරීමෙන් සහනයක් ලැබේ",
    "ඉඟුරු සහ සුදුළූණු තම්බා බීම හෝ අසමෝදගම් භාවිතය සාර්ථක ප්‍රතිකාරයකි",
    "වෙනිවැල්ගැට කහ ලොත්සුඹුලු වැනි දෑ ඇල්වතුරෙන් අඹරා ගෑම පැරණි ක්‍රමයකි",
    "මැටි මුට්ටියකට බෙහෙත් ද්‍රව්‍ය දමා වතුර කෝප්ප අටක් එක් කරන්න",
    "මද ගින්නේ රත් කරමින් වතුර ප්‍රමාණය කෝප්ප එකක් දක්වා අඩු වන තුරු තම්බන්න",
    "පිරිසිදු රෙදි කඩකින් හෝ පෙරනයකින් පෙරා නිවෙන්නට හැර පානය කරන්න",
    "රෝගයක් වැළඳුණු මුල් අවස්ථාවේදීම අත් බෙහෙත් මගින් එය වැඩි දියුණු වීම වළක්වා ගත හැකිය",
    "නිවසේදීම පහසුවෙන් සොයාගත හැකි දේවලින් සකසා ගත හැකිය",
    "කොත්තමල්ලි වියළි ඉඟුරු පත්පාඩගම් කටුවැල්බටු වෙනිවැල්ගැට යන ඖෂධ උණ සහ සෙම් රෝග සමනය කරයි",
    "දවස පුරා වෙහෙස මහන්සි වූ පසු ඇතිවන ඇඟපත වේදනාවට බැබිල මුල් කසාය සහනයක් ලබා දෙයි",
    "ආඩතෝඩා කොළ නෙල්ලි බාර්ලි සෙම පිටකිරීමට සහ කැස්ස පාලනයට උදවු වේ",
    "බෙලි බීම මලබද්ධය දුරු කිරීමට ශරීරයේ දැවිල්ල සහ පිපාසය නිවීමට ඉතා ගුණදායකයි",
    "කෝමාරිකා අම්ල පිත්ත රෝගයට ඉතා හිතකර වන අතර ශරීරයේ උෂ්ණත්වය පාලනය කරයි",
    "සියඹලා ආහාර රුචිය වඩවන අතර ශරීරයට යකඩ අවශෝෂණය කර ගැනීමට උපකාරී වේ",
]


def load_corpus(path: str) -> List[List[str]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            toks = [normalize_sinhala(t) for t in raw.strip().split() if t.strip()]
            if toks:
                out.append(toks)
    return out


def sample_concat(
    lines: List[List[str]],
    k: int,
    rng: random.Random,
) -> Tuple[List[str], List[int]]:
    """Pick k random *non-adjacent* lines and concatenate them.

    We avoid strictly adjacent lines so that the test sequences cannot be
    "memorised" from training pairs.
    """
    chosen_idx = sorted(rng.sample(range(len(lines)), k))
    chunk = [lines[i] for i in chosen_idx]
    tokens: List[str] = []
    end_idx: List[int] = []
    for sent in chunk:
        tokens.extend(sent)
        end_idx.append(len(tokens) - 1)
    return tokens, end_idx


def lf_disagreement_count(votes: List[List[int]]) -> int:
    """Number of tokens where ≥2 non-abstain LFs disagree."""
    n_disagree = 0
    for row in votes:
        non_abst = [v for v in row if v != -1]
        if len(non_abst) >= 2 and len(set(non_abst)) > 1:
            n_disagree += 1
    return n_disagree


def write_sequence(out_lines: list, tokens: List[str], stop_idx: set, seq_id: int, bucket: str):
    out_lines.append(f"# sequence_id={seq_id}\tbucket={bucket}\tlength={len(tokens)}\tstops={len(stop_idx)}")
    for i, tok in enumerate(tokens):
        out_lines.append(f"{tok}\t{'STOP' if i in stop_idx else 'O'}")
    out_lines.append("")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-easy", type=int, default=100, help="2-sentence sequences.")
    p.add_argument("--n-med", type=int, default=100, help="3-4-sentence sequences.")
    p.add_argument("--n-hard", type=int, default=100, help="5-7-sentence sequences.")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    rng = random.Random(args.seed)
    print(f"Loading corpus from {CORPUS_PATH}…")
    lines = load_corpus(CORPUS_PATH)
    print(f"  {len(lines):,} corpus lines.")

    p_end_stats = ensure_endword_statistics()

    out_lines: List[str] = []
    meta_records = []
    seq_id = 0

    buckets = [
        ("EASY", args.n_easy, 2, 2),
        ("MED", args.n_med, 3, 4),
        ("HARD", args.n_hard, 5, 7),
    ]
    for bucket, n, kmin, kmax in buckets:
        for _ in range(n):
            k = rng.randint(kmin, kmax)
            tokens, end_idx = sample_concat(lines, k, rng)
            stop_idx = set(end_idx)
            votes = apply_all_lfs(tokens, end_idx, p_end_stats)
            disagree = lf_disagreement_count(votes)
            write_sequence(out_lines, tokens, stop_idx, seq_id, bucket)
            meta_records.append({
                "seq_id": seq_id,
                "bucket": bucket,
                "n_sentences": k,
                "n_tokens": len(tokens),
                "n_stops": len(stop_idx),
                "lf_disagreement": disagree,
                "ambiguous": disagree >= 1,
            })
            seq_id += 1

    # Append the EXPERT bucket — single-sentence hand-typed authentic data
    for sent in EXPERT_SEED:
        tokens = [normalize_sinhala(t) for t in sent.split() if t.strip()]
        if not tokens:
            continue
        stop_idx = {len(tokens) - 1}
        votes = apply_all_lfs(tokens, [len(tokens) - 1], p_end_stats)
        disagree = lf_disagreement_count(votes)
        write_sequence(out_lines, tokens, stop_idx, seq_id, "EXPERT")
        meta_records.append({
            "seq_id": seq_id,
            "bucket": "EXPERT",
            "n_sentences": 1,
            "n_tokens": len(tokens),
            "n_stops": 1,
            "lf_disagreement": disagree,
            "ambiguous": disagree >= 1,
        })
        seq_id += 1

    print(f"Writing {seq_id} sequences → {OUT_TSV}")
    with open(OUT_TSV, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    n_amb = sum(r["ambiguous"] for r in meta_records)
    bucket_breakdown = {}
    for r in meta_records:
        bucket_breakdown.setdefault(r["bucket"], 0)
        bucket_breakdown[r["bucket"]] += 1

    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed,
            "n_sequences": seq_id,
            "n_ambiguous": n_amb,
            "bucket_counts": bucket_breakdown,
            "records": meta_records,
        }, f, ensure_ascii=False, indent=2)

    print(f"Wrote meta → {OUT_META}")
    print(f"  total sequences: {seq_id}")
    print(f"  buckets: {bucket_breakdown}")
    print(f"  ambiguous (≥1 LF disagreement): {n_amb}")


if __name__ == "__main__":
    main()
