"""
Build the Phase-2 KG-Safety benchmark (v2).

Why we need this
----------------
The Phase-1 safety evaluation in ``evaluation/evaluate.py`` uses 8 hard-coded
test strings. That is insufficient for any clinical-safety claim
(``docs/FULL_AUDIT_REPORT.md`` §2.5).

This script *programmatically* constructs ~70 multi-sentence Ayurvedic-recipe
scenarios that stress every safety-relevant axis:

    SAFE_HERB           ─ only low-toxicity herbs        → APPROVE
    TOXIC_PURIFIED      ─ toxic + correct shodhana       → APPROVE
    TOXIC_ADJACENT      ─ toxic in sent A, shodhana in A+1
                           tests sliding-window k≥1     → APPROVE if k≥1
    TOXIC_DISTANT       ─ toxic in sent A, shodhana 3 sents away
                           tests window k≥3              → APPROVE if k≥3 else REJECT
    TOXIC_UNPURIFIED    ─ toxic, no shodhana anywhere    → REJECT
    TOXIC_NEGATED       ─ toxic + "නොකරන්න" / "එපා"     → REJECT
    MULTI_PARTIAL       ─ ≥2 toxic; some purified        → REJECT
    MULTI_ALL_PURIFIED  ─ ≥2 toxic; all purified         → APPROVE
    ALIAS_PURIFIED      ─ toxic referenced by alias name → APPROVE
    PLANTPART_PURIFIED  ─ "X කොළ" with shodhana          → APPROVE

For every scenario the script emits:
    * ``text``             — segmented text ready for ``analyze_safety``
    * ``expected_verdict`` — APPROVE / REJECT
    * ``expected_at_window`` — dict mapping context-window k to verdict
    * ``scenario_kind``    — label from the list above

Output: ``data/safety_benchmark.jsonl``
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import DATA_DIR, normalize_sinhala  # noqa: E402

KG_PATH = os.path.join(DATA_DIR, "ayurvedic_ingredients_full.csv")
OUT_JSONL = os.path.join(DATA_DIR, "safety_benchmark.jsonl")

NEGATION_PHRASES = ["නොකරන්න", "එපා", "නොකර"]
SAFE_HERBS = [
    "ඉඟුරු", "කොත්තමල්ලි", "පත්පාඩගම්", "කටුවැල්බටු",
    "කහ", "වෙනිවැල්ගැට", "තිප්පිලි", "රසකිඳ",
]

# Template sentences (each template is a *single* sentence already segmented
# with a trailing period when emitted via emit_text). {TOX} and {SHO} are
# filled by the scenario builder.
TPL_INTRO = [
    "පැරණි බෙහෙත් ක්‍රමය මෙසේ පැවසේ.",
    "ආයුර්වේද ග්‍රන්ථයන්හි මෙසේ සඳහන් වේ.",
    "මේ අත් බෙහෙත සිංහල වෛද්‍ය ක්‍රමයේ පවතී.",
]
TPL_TOX_USE = [
    "{TOX} භාවිතය කළ හැකිය.",
    "මෙම ඖෂධයට {TOX} ද ඇතුළත් කරනු.",
    "{TOX} මුල් සහ පොතු ගෙන කෂාය කරනු.",
]
TPL_SHO = [
    "ප්‍රථමයෙන් {SHO} සමඟ එය පිරිසිදු කරගත යුතුය.",
    "{SHO} සමඟ තම්බා පිරිසිදු කරගනු.",
    "{SHO} ක්‍රමයෙන් ශෝධනය කිරීම අවශ්‍යයි.",
]
TPL_DIRECT_PURE = [
    "{TOX} {SHO} සමඟ පිරිසිදු කර කෂාය කරනු.",
    "ශුද්ධ කරගත් {TOX} {SHO} සමඟ මිශ්‍ර කර තබනු.",
]
TPL_FILLER = [
    "මී පැණි සමඟ දිවා ආහාරයෙන් පසු සේවනය කරනු.",
    "උදේ සවස වරක් මිනිත්තු තිහක් පමණ පානය කරනු.",
    "ලඟ ඇති සියළු ද්‍රව්‍යයන් අඹරා කල්වරක් තබනු.",
    "මේ ඖෂධය වයස අවුරුදු දොළහට වැඩි අයට පමණි.",
]
TPL_OUTRO = [
    "මෙය දින හතක් අඛණ්ඩව ලබා දෙනු.",
    "ප්‍රතිඵලය ලැබෙන තුරු සියලු උපදෙස් පිළිපදිනු.",
]


def load_kg(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "entity": normalize_sinhala(r["Entity"].strip()),
                "aliases": [normalize_sinhala(a.strip()) for a in r["Aliases"].split(",") if a.strip()],
                "toxicity": r["Toxicity"].strip().lower(),
                "shodhana": [normalize_sinhala(p.strip()) for p in r["Purification_Keywords"].split(",") if p.strip()],
            })
    return rows


def pick_high_toxic(kg: List[dict]) -> List[dict]:
    base_entities = [k for k in kg if k["toxicity"] in {"high", "medium-high"} and " " not in k["entity"]]
    return base_entities


def join_text(sentences: List[str]) -> str:
    """Sentences are already terminated with a full stop; join with spaces."""
    return " ".join(sentences)


def emit(scenarios: list, *, scenario_kind: str, sentences: List[str],
         expected_verdict: str, expected_at_window: Dict[int, str], notes: str = ""):
    scenarios.append({
        "scenario_kind": scenario_kind,
        "text": join_text(sentences),
        "expected_verdict": expected_verdict,
        "expected_at_window": {str(k): v for k, v in expected_at_window.items()},
        "notes": notes,
    })


def build_safe_herb(rng: random.Random) -> list:
    s = []
    s.append(rng.choice(TPL_INTRO))
    chosen = rng.sample(SAFE_HERBS, k=3)
    s.append(f"{chosen[0]} සහ {chosen[1]} සමානව ගෙන තම්බා {chosen[2]} සමඟ පානය කරනු.")
    s.append(rng.choice(TPL_FILLER))
    s.append(rng.choice(TPL_OUTRO))
    return s


def build_toxic_purified(rng: random.Random, tox: dict) -> list:
    sho = rng.choice(tox["shodhana"])
    s = []
    s.append(rng.choice(TPL_INTRO))
    s.append(rng.choice(TPL_DIRECT_PURE).format(TOX=tox["entity"], SHO=sho))
    s.append(rng.choice(TPL_FILLER))
    return s


def build_toxic_adjacent(rng: random.Random, tox: dict) -> list:
    sho = rng.choice(tox["shodhana"])
    s = []
    s.append(rng.choice(TPL_INTRO))
    s.append(rng.choice(TPL_TOX_USE).format(TOX=tox["entity"]))
    s.append(rng.choice(TPL_SHO).format(SHO=sho))
    s.append(rng.choice(TPL_FILLER))
    return s


def build_toxic_distant(rng: random.Random, tox: dict) -> list:
    sho = rng.choice(tox["shodhana"])
    s = []
    s.append(rng.choice(TPL_INTRO))
    s.append(rng.choice(TPL_TOX_USE).format(TOX=tox["entity"]))
    # 3 filler sentences between toxic mention and shodhana
    for _ in range(3):
        s.append(rng.choice(TPL_FILLER))
    s.append(rng.choice(TPL_SHO).format(SHO=sho))
    s.append(rng.choice(TPL_OUTRO))
    return s


def build_toxic_unpurified(rng: random.Random, tox: dict) -> list:
    s = []
    s.append(rng.choice(TPL_INTRO))
    s.append(rng.choice(TPL_TOX_USE).format(TOX=tox["entity"]))
    s.append(rng.choice(TPL_FILLER))
    s.append(rng.choice(TPL_OUTRO))
    return s


def build_toxic_negated(rng: random.Random, tox: dict) -> list:
    sho = rng.choice(tox["shodhana"])
    neg = rng.choice(NEGATION_PHRASES)
    s = []
    s.append(rng.choice(TPL_INTRO))
    s.append(rng.choice(TPL_TOX_USE).format(TOX=tox["entity"]))
    s.append(f"{sho} සමඟ පිරිසිදු කිරීම {neg}.")
    s.append(rng.choice(TPL_OUTRO))
    return s


def build_multi_partial(rng: random.Random, toxics: List[dict]) -> list:
    a, b = rng.sample(toxics, 2)
    sho_a = rng.choice(a["shodhana"])
    s = []
    s.append(rng.choice(TPL_INTRO))
    s.append(rng.choice(TPL_DIRECT_PURE).format(TOX=a["entity"], SHO=sho_a))
    s.append(rng.choice(TPL_TOX_USE).format(TOX=b["entity"]))
    s.append(rng.choice(TPL_FILLER))
    return s


def build_multi_all_purified(rng: random.Random, toxics: List[dict]) -> list:
    a, b = rng.sample(toxics, 2)
    sho_a = rng.choice(a["shodhana"])
    sho_b = rng.choice(b["shodhana"])
    s = []
    s.append(rng.choice(TPL_INTRO))
    s.append(rng.choice(TPL_DIRECT_PURE).format(TOX=a["entity"], SHO=sho_a))
    s.append(rng.choice(TPL_DIRECT_PURE).format(TOX=b["entity"], SHO=sho_b))
    return s


def build_alias_purified(rng: random.Random, tox: dict) -> list:
    if not tox["aliases"]:
        return build_toxic_purified(rng, tox)
    alias = rng.choice(tox["aliases"])
    sho = rng.choice(tox["shodhana"])
    s = []
    s.append(rng.choice(TPL_INTRO))
    s.append(f"{alias} {sho} සමඟ පිරිසිදු කර ගෙන භාවිතය කරනු.")
    s.append(rng.choice(TPL_FILLER))
    return s


def build_plantpart_purified(rng: random.Random, tox: dict) -> list:
    parts = ["කොළ", "මුල්", "පොතු", "මල්", "ඇට"]
    part = rng.choice(parts)
    sho = rng.choice(tox["shodhana"])
    s = []
    s.append(rng.choice(TPL_INTRO))
    s.append(f"{tox['entity']} {part} {sho} සමඟ පිරිසිදු කර කෂාය කරනු.")
    s.append(rng.choice(TPL_FILLER))
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-kind", type=int, default=7,
                   help="Number of scenarios per scenario_kind.")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    rng = random.Random(args.seed)
    kg = load_kg(KG_PATH)
    toxics = pick_high_toxic(kg)
    if len(toxics) < 2:
        raise RuntimeError("Need ≥2 base toxic entities in KG.")

    scenarios = []

    def add(kind: str, n: int, builder, verdict: str, win: Dict[int, str]):
        for _ in range(n):
            if kind in ("MULTI_PARTIAL", "MULTI_ALL_PURIFIED"):
                sents = builder(rng, toxics)
            elif kind == "SAFE_HERB":
                sents = builder(rng)
            else:
                tox = rng.choice(toxics)
                sents = builder(rng, tox)
            emit(scenarios, scenario_kind=kind, sentences=sents,
                 expected_verdict=verdict, expected_at_window=win)

    add("SAFE_HERB",        args.per_kind, build_safe_herb,        "APPROVE",
        {0: "APPROVE", 1: "APPROVE", 2: "APPROVE", 3: "APPROVE"})
    add("TOXIC_PURIFIED",   args.per_kind, build_toxic_purified,   "APPROVE",
        {0: "APPROVE", 1: "APPROVE", 2: "APPROVE", 3: "APPROVE"})
    add("TOXIC_ADJACENT",   args.per_kind, build_toxic_adjacent,   "APPROVE",
        {0: "REJECT",  1: "APPROVE", 2: "APPROVE", 3: "APPROVE"})
    add("TOXIC_DISTANT",    args.per_kind, build_toxic_distant,    "REJECT",
        {0: "REJECT",  1: "REJECT",  2: "REJECT",  3: "APPROVE"})
    add("TOXIC_UNPURIFIED", args.per_kind, build_toxic_unpurified, "REJECT",
        {0: "REJECT",  1: "REJECT",  2: "REJECT",  3: "REJECT"})
    add("TOXIC_NEGATED",    args.per_kind, build_toxic_negated,    "REJECT",
        {0: "REJECT",  1: "REJECT",  2: "REJECT",  3: "REJECT"})
    add("MULTI_PARTIAL",    args.per_kind, build_multi_partial,    "REJECT",
        {0: "REJECT",  1: "REJECT",  2: "REJECT",  3: "REJECT"})
    add("MULTI_ALL_PURIFIED", args.per_kind, build_multi_all_purified, "APPROVE",
        {0: "APPROVE", 1: "APPROVE", 2: "APPROVE", 3: "APPROVE"})
    add("ALIAS_PURIFIED",   args.per_kind, build_alias_purified,   "APPROVE",
        {0: "APPROVE", 1: "APPROVE", 2: "APPROVE", 3: "APPROVE"})
    add("PLANTPART_PURIFIED", args.per_kind, build_plantpart_purified, "APPROVE",
        {0: "APPROVE", 1: "APPROVE", 2: "APPROVE", 3: "APPROVE"})

    print(f"Generated {len(scenarios)} scenarios → {OUT_JSONL}")
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Quick distribution summary
    from collections import Counter
    by_kind = Counter(s["scenario_kind"] for s in scenarios)
    for k, n in by_kind.items():
        print(f"  {k}: {n}")


if __name__ == "__main__":
    main()
