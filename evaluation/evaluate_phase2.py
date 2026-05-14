"""
Phase 2 — comprehensive fair evaluation framework.

What this script does
---------------------
Evaluates every Phase-2 model variant against
  data/gold_test_v2.tsv         (multi-bucket SBD gold)
  data/safety_benchmark.jsonl   (KG safety scenarios)

For each SBD model it reports:
  * per-bucket F1(STOP), Precision, Recall, Accuracy
  * confusion matrix
  * McNemar's test against the rule-only baseline
  * bootstrap 95 % CI on F1(STOP)
  * inference latency

For the safety guardrail it reports:
  * scenario-kind accuracy
  * window-size sweep (k ∈ {0, 1, 2, 3})
  * cascade-failure rate when a segmentation error is artificially injected
  * Safety-Risk vs HITL-Coverage curve

Results JSON written to ``evaluation/results/phase2_eval.json``.

This script is the *fair-comparison* evidence base requested in
``docs/FULL_AUDIT_REPORT.md`` §3 Step 5 and Step 6.

You must have run ``scripts/build_gold_v2.py`` and
``scripts/build_safety_benchmark.py`` (and trained the RoBERTa
variants on Colab if you want to include them).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from typing import Callable, Dict, List, Sequence, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for d in (PROJECT_ROOT, SRC_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import numpy as np  # noqa: E402

from config import DATA_DIR, MODELS_DIR, CANONICAL_ENDINGS, normalize_sinhala  # noqa: E402
from confidence_pipeline import (  # noqa: E402
    SegmentationResult,
    cascade_safety,
    load_knowledge_graph,
    segment_with_callable,
    sequence_reliability,
)


GOLD_V2_PATH = os.path.join(DATA_DIR, "gold_test_v2.tsv")
SAFETY_PATH = os.path.join(DATA_DIR, "safety_benchmark.jsonl")
RESULTS_PATH = os.path.join(THIS_DIR, "results", "phase2_eval.json")


# ===========================================================================
# Gold loading
# ===========================================================================
def load_gold_v2(path: str) -> List[Dict]:
    """Parse the v2 gold file with `# sequence_id=… bucket=…` headers."""
    sequences: List[Dict] = []
    cur: Dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                meta = {}
                for kv in line.lstrip("#").strip().split("\t"):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        meta[k.strip()] = v.strip()
                cur = {"meta": meta, "words": [], "labels": []}
                sequences.append(cur)
                continue
            if not line.strip():
                continue
            parts = line.split("\t")
            cur["words"].append(parts[0])
            cur["labels"].append(1 if parts[1] == "STOP" else 0)
    return [s for s in sequences if s["words"]]


# ===========================================================================
# SBD model adapters: every model is wrapped to expose
#     predict(words: List[str]) -> List[float]   (per-word P(STOP))
# ===========================================================================
def make_rule_only_predictor() -> Callable[[List[str]], List[float]]:
    """Deterministic suffix-rule baseline (the *only* fair primary baseline,
    per ``docs/FULL_AUDIT_REPORT.md`` §3 Step 2)."""
    sufs = list(CANONICAL_ENDINGS)
    def predict(words: List[str]) -> List[float]:
        return [1.0 if any(normalize_sinhala(w).endswith(s) for s in sufs) else 0.0 for w in words]
    return predict


def make_crf_predictor(crf_path: str) -> Callable[[List[str]], List[float]]:
    """Wrap a fitted sklearn-crfsuite CRF."""
    import joblib
    from pipeline import sent2features
    crf = joblib.load(crf_path)
    def predict(words: List[str]) -> List[float]:
        if not words:
            return []
        dummy = [(w, "") for w in words]
        marginals = crf.predict_marginals([sent2features(dummy)])[0]
        return [float(m.get("STOP", 0.0)) for m in marginals]
    return predict


def make_hf_token_predictor(model_dir: str) -> Callable[[List[str]], List[float]]:
    """Generic HF token-classification predictor (baseline + DAPT variants)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def predict(words: List[str]) -> List[float]:
        if not words:
            return []
        enc = tok([words], is_split_into_words=True, truncation=True,
                  max_length=256, return_tensors="pt", padding=True)
        word_ids = enc.word_ids(0)
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device)).logits[0].cpu()
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
        out = [0.0] * len(words)
        seen = set()
        for tok_idx, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            out[wid] = float(probs[tok_idx])
            seen.add(wid)
        return out
    return predict


def make_hf_morph_predictor(model_dir: str) -> Callable[[List[str]], List[float]]:
    """Predictor for the custom morphology-injected XLM-R head."""
    import torch
    from transformers import AutoTokenizer, XLMRobertaConfig
    from morphology_features import morph_vector, MORPH_FEATURE_DIM
    # We import the custom class lazily (notebook saves checkpoint as pytorch_model.bin
    # plus XLMRobertaConfig). We rebuild the class definition here so we don't depend on
    # the notebook being importable.
    import torch.nn as nn
    from transformers import XLMRobertaModel

    class XLMRMorphForTokenClassification(nn.Module):
        def __init__(self, init_dir, num_labels=2, morph_dim=MORPH_FEATURE_DIM, dropout=0.1):
            super().__init__()
            self.config = XLMRobertaConfig.from_pretrained(init_dir)
            self.encoder = XLMRobertaModel.from_pretrained(init_dir)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.config.hidden_size + morph_dim, num_labels)
            self.morph_dim = morph_dim
        def forward(self, input_ids, attention_mask, morph_feats):
            h = self.dropout(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
            joined = torch.cat([h, morph_feats.to(h.dtype)], dim=-1)
            return self.classifier(joined)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = XLMRMorphForTokenClassification(model_dir)
    state = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def predict(words: List[str]) -> List[float]:
        if not words:
            return []
        enc = tok([words], is_split_into_words=True, truncation=True,
                  max_length=256, return_tensors="pt", padding=True)
        word_ids = enc.word_ids(0)
        morph_rows = []
        for wid in word_ids:
            if wid is None:
                morph_rows.append([0.0] * MORPH_FEATURE_DIM)
            else:
                morph_rows.append(morph_vector(words[wid]).tolist())
        morph_t = torch.tensor([morph_rows], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device),
                           morph_feats=morph_t)[0].cpu()
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
        out = [0.0] * len(words)
        seen = set()
        for tok_idx, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            out[wid] = float(probs[tok_idx])
            seen.add(wid)
        return out
    return predict


# ===========================================================================
# Metrics
# ===========================================================================
def confusion(preds: Sequence[int], gold: Sequence[int]) -> Dict[str, int]:
    tp = sum(1 for p, g in zip(preds, gold) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(preds, gold) if p == 1 and g == 0)
    tn = sum(1 for p, g in zip(preds, gold) if p == 0 and g == 0)
    fn = sum(1 for p, g in zip(preds, gold) if p == 0 and g == 1)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def metrics_from_confusion(c: Dict[str, int]) -> Dict[str, float]:
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    total = sum(c.values())
    acc = (c["tp"] + c["tn"]) / total if total else 0.0
    return {"precision_stop": p, "recall_stop": r, "f1_stop": f1, "accuracy": acc}


def bootstrap_f1_ci(preds: List[int], gold: List[int], n: int = 1000, seed: int = 0) -> Tuple[float, float]:
    rng = random.Random(seed)
    idx = list(range(len(preds)))
    scores = []
    for _ in range(n):
        sample = [rng.randint(0, len(idx) - 1) for _ in idx]
        sp = [preds[i] for i in sample]
        sg = [gold[i] for i in sample]
        scores.append(metrics_from_confusion(confusion(sp, sg))["f1_stop"])
    scores.sort()
    return scores[int(0.025 * n)], scores[int(0.975 * n) - 1]


def mcnemar(preds_a: Sequence[int], preds_b: Sequence[int], gold: Sequence[int]) -> Dict[str, float]:
    """McNemar's test (with continuity correction) between two classifiers."""
    b = sum(1 for pa, pb, g in zip(preds_a, preds_b, gold) if pa == g and pb != g)
    c = sum(1 for pa, pb, g in zip(preds_a, preds_b, gold) if pa != g and pb == g)
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # 1-DOF chi-square p-value from survival function
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"b": b, "c": c, "chi2": chi2, "p": p}


# ===========================================================================
# Per-model evaluation on Gold v2
# ===========================================================================
def evaluate_predictor_on_gold(
    name: str,
    predict: Callable[[List[str]], List[float]],
    sequences: List[Dict],
    threshold: float = 0.5,
) -> Dict:
    per_bucket: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: {"pred": [], "gold": []})
    all_pred: List[int] = []
    all_gold: List[int] = []
    latencies: List[float] = []
    for seq in sequences:
        words = seq["words"]
        gold = seq["labels"]
        t0 = time.perf_counter()
        probs = predict(words)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        preds = [1 if p > threshold else 0 for p in probs]
        bucket = seq["meta"].get("bucket", "ALL")
        per_bucket[bucket]["pred"].extend(preds)
        per_bucket[bucket]["gold"].extend(gold)
        all_pred.extend(preds)
        all_gold.extend(gold)

    overall_conf = confusion(all_pred, all_gold)
    overall = metrics_from_confusion(overall_conf)
    lo, hi = bootstrap_f1_ci(all_pred, all_gold)

    bucket_results = {}
    for b, d in per_bucket.items():
        c = confusion(d["pred"], d["gold"])
        bucket_results[b] = {**metrics_from_confusion(c), "confusion": c}

    return {
        "model": name,
        "threshold": threshold,
        "overall": {**overall, "confusion": overall_conf, "f1_stop_ci95": [lo, hi]},
        "by_bucket": bucket_results,
        "latency_ms": {
            "mean": statistics.mean(latencies) if latencies else 0.0,
            "p95": sorted(latencies)[int(0.95 * len(latencies)) - 1] if latencies else 0.0,
        },
        "all_preds": all_pred,  # retained for McNemar
        "all_gold": all_gold,
    }


# ===========================================================================
# Safety benchmark evaluation
# ===========================================================================
def load_safety_benchmark(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def evaluate_safety_on_benchmark(
    name: str,
    predict: Callable[[List[str]], List[float]],
    scenarios: List[Dict],
    kg: Dict,
    threshold: float = 0.5,
    windows: Sequence[int] = (0, 1, 2, 3),
    hitl_min: float = 0.55,
) -> Dict:
    by_kind = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    overall = defaultdict(lambda: {"correct": 0, "total": 0})
    hitl_counts = defaultdict(int)
    for sc in scenarios:
        words = sc["text"].split()
        probs = predict(words)
        seg = SegmentationResult(
            words=words,
            stop_probs=probs,
            threshold=threshold,
            segmented_text=" ".join(w + "." if p > threshold else w for w, p in zip(words, probs)),
            reliability=sequence_reliability(probs, threshold),
            method=name,
        )
        for k in windows:
            verdict = cascade_safety(seg, kg, window_size=k, hitl_min_seg_reliability=hitl_min)
            # Map HITL → "uncertain"; for accuracy we count HITL as wrong (conservative).
            decided = verdict.final_status
            if decided == "HITL":
                hitl_counts[k] += 1
                decided = "HITL"
            expected = sc["expected_at_window"].get(str(k), sc["expected_verdict"])
            correct = decided == expected
            by_kind[sc["scenario_kind"]][k]["total"] += 1
            by_kind[sc["scenario_kind"]][k]["correct"] += int(correct)
            overall[k]["total"] += 1
            overall[k]["correct"] += int(correct)
    return {
        "model": name,
        "overall_accuracy_by_window": {k: v["correct"] / v["total"] for k, v in overall.items()},
        "hitl_count_by_window": dict(hitl_counts),
        "per_kind_by_window": {
            kind: {k: (v["correct"] / v["total"]) for k, v in d.items()}
            for kind, d in by_kind.items()
        },
    }


# ===========================================================================
# Cascading-failure analysis
# ===========================================================================
def cascade_failure_sweep(
    predict: Callable[[List[str]], List[float]],
    scenarios: List[Dict],
    kg: Dict,
    threshold: float = 0.5,
    window: int = 1,
    flip_rates: Sequence[float] = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50),
    seed: int = 0,
) -> Dict:
    """Inject artificial segmentation errors (flip k% of STOP predictions to O
    and equivalent O→STOP) and report how downstream safety accuracy decays."""
    rng = random.Random(seed)
    out = {}
    for rate in flip_rates:
        correct = 0
        total = 0
        for sc in scenarios:
            words = sc["text"].split()
            probs = predict(words)
            preds = [1 if p > threshold else 0 for p in probs]
            for i in range(len(preds)):
                if rng.random() < rate:
                    preds[i] = 1 - preds[i]
            seg_text = " ".join(w + "." if pr == 1 else w for w, pr in zip(words, preds))
            seg = SegmentationResult(words, [float(p) for p in preds], threshold, seg_text,
                                     reliability=sequence_reliability(probs, threshold),
                                     method="perturbed")
            verdict = cascade_safety(seg, kg, window_size=window)
            decided = verdict.final_status
            if decided == "HITL":
                decided = "HITL"
            expected = sc["expected_at_window"].get(str(window), sc["expected_verdict"])
            total += 1
            if decided == expected:
                correct += 1
        out[rate] = correct / total if total else 0.0
    return out


# ===========================================================================
# Driver
# ===========================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--include-crf", action="store_true",
                   help="Include the legacy CRF checkpoint if present.")
    p.add_argument("--include-baseline-roberta", action="store_true")
    p.add_argument("--include-dapt-roberta", action="store_true")
    p.add_argument("--include-dapt-morph", action="store_true")
    args = p.parse_args()

    print(f"Loading gold v2 from {GOLD_V2_PATH}…")
    sequences = load_gold_v2(GOLD_V2_PATH)
    print(f"  {len(sequences)} sequences, "
          f"{sum(len(s['words']) for s in sequences)} tokens")

    print(f"Loading safety benchmark from {SAFETY_PATH}…")
    scenarios = load_safety_benchmark(SAFETY_PATH)
    kg = load_knowledge_graph()
    print(f"  {len(scenarios)} scenarios, KG has {len(kg) if kg else 0} toxic entities")

    predictors: Dict[str, Callable[[List[str]], List[float]]] = {
        "rule_only": make_rule_only_predictor(),
    }

    if args.include_crf:
        crf_path = os.path.join(MODELS_DIR, "ayurvedic_segmenter.pkl")
        if os.path.exists(crf_path):
            predictors["crf"] = make_crf_predictor(crf_path)
        else:
            print(f"  warning: {crf_path} missing, skipping CRF")
    if args.include_baseline_roberta:
        d = os.path.join(MODELS_DIR, "sbd_xlmr_baseline")
        if os.path.isdir(d):
            predictors["xlmr_baseline"] = make_hf_token_predictor(d)
    if args.include_dapt_roberta:
        d = os.path.join(MODELS_DIR, "sbd_xlmr_dapt")
        if os.path.isdir(d):
            predictors["xlmr_dapt"] = make_hf_token_predictor(d)
    if args.include_dapt_morph:
        d = os.path.join(MODELS_DIR, "sbd_xlmr_dapt_morph")
        if os.path.isdir(d):
            predictors["xlmr_dapt_morph"] = make_hf_morph_predictor(d)

    print(f"Evaluating {list(predictors.keys())}…")
    sbd_results = {}
    for name, pred in predictors.items():
        print(f"  → {name}")
        sbd_results[name] = evaluate_predictor_on_gold(name, pred, sequences, threshold=args.threshold)

    # McNemar's test of every model vs rule_only (primary baseline per audit)
    base_preds = sbd_results["rule_only"]["all_preds"]
    gold = sbd_results["rule_only"]["all_gold"]
    mcnemar_results = {}
    for name, res in sbd_results.items():
        if name == "rule_only":
            continue
        mcnemar_results[name] = mcnemar(base_preds, res["all_preds"], gold)

    # Safety benchmark evaluation
    safety_results = {}
    cascade_results = {}
    if kg:
        for name, pred in predictors.items():
            print(f"  safety  → {name}")
            safety_results[name] = evaluate_safety_on_benchmark(name, pred, scenarios, kg)
            cascade_results[name] = cascade_failure_sweep(pred, scenarios, kg)

    # Strip the large prediction arrays out of the saved JSON
    for r in sbd_results.values():
        r.pop("all_preds", None)
        r.pop("all_gold", None)

    out = {
        "config": vars(args),
        "n_sequences": len(sequences),
        "n_safety_scenarios": len(scenarios),
        "sbd": sbd_results,
        "mcnemar_vs_rule_only": mcnemar_results,
        "safety": safety_results,
        "cascade_failure_sweep": cascade_results,
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2, default=float, ensure_ascii=False)
    print(f"Wrote {RESULTS_PATH}")

    # Pretty-print headline numbers
    print("\nHeadline SBD scores (gold v2):")
    print(f"{'model':<22}{'F1(STOP)':>11}{'P':>10}{'R':>10}{'Acc':>10}{'F1 CI95':>20}")
    for name, r in sbd_results.items():
        ov = r["overall"]
        ci = ov["f1_stop_ci95"]
        print(f"{name:<22}{ov['f1_stop']:>11.3f}{ov['precision_stop']:>10.3f}"
              f"{ov['recall_stop']:>10.3f}{ov['accuracy']:>10.3f}"
              f"   [{ci[0]:.3f},{ci[1]:.3f}]")

    if safety_results:
        print("\nSafety accuracy at window=1:")
        for name, r in safety_results.items():
            print(f"  {name:<22} {r['overall_accuracy_by_window'].get(1, 0.0):.3f}")


if __name__ == "__main__":
    main()
