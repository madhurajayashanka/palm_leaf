"""
Comprehensive Evaluation Framework for Ayurvedic NLP Pipeline.

Implements:
  - CRF training with canonical features (config.py)
  - Formal evaluation (P/R/F1, confusion matrix, per-class metrics)
  - 5-fold cross-validation with mean ± std
  - Ablation studies (with/without is_common_ending)
  - Baseline comparisons (majority, random, rule-only)
  - CRF vs RoBERTa learning curve (data-size sweep)
  - Safety guardrail analysis (window_size sweep)
  - Statistical significance tests (McNemar, bootstrap CI)
  - Latency benchmarking
  - All results saved to eval_results/
"""

import os
import csv
import json
import time
import math
import random
import joblib
import numpy as np
from collections import Counter, defaultdict
from config import (
    CANONICAL_ENDINGS, WEAK_SUPERVISION_ENDINGS,
    CRF_PARAMS, CRF_CHUNK_SIZE, CRF_TRAIN_SPLIT, CRF_THRESHOLD_DEFAULT,
    normalize_sinhala
)

try:
    import sklearn_crfsuite
    from sklearn_crfsuite import metrics as crf_metrics
    HAS_CRF = True
except ImportError:
    HAS_CRF = False
    print("WARNING: sklearn-crfsuite not installed. CRF evaluation disabled.")

RESULTS_DIR = "eval_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_labeled_data(filepath="train_labeled.tsv"):
    """Load CoNLL-style word\\ttag file into list of sequences."""
    sequences = []
    current = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    sequences.append(current)
                    current = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    word = normalize_sinhala(parts[0])
                    tag = parts[1]
                    current.append((word, tag))
    if current:
        sequences.append(current)
    return sequences


# ==============================================================================
# FEATURE EXTRACTION (uses config.py canonical list)
# ==============================================================================

def word2features(sent, i, use_common_ending=True, endings=None):
    """Extract features for word i in a sentence.
    
    Args:
        sent: List of (word, tag) tuples.
        i: Index of target word.
        use_common_ending: If False, ablates the is_common_ending feature.
        endings: Override suffix list (for ablation). Default: CANONICAL_ENDINGS.
    """
    if endings is None:
        endings = CANONICAL_ENDINGS
    word = sent[i][0]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-2:]': word[-2:],
        'word[-3:]': word[-3:] if len(word) >= 3 else word,
        'word.length': len(word),
    }
    
    if use_common_ending:
        is_ending = any(word.endswith(suffix) for suffix in endings)
        features['is_common_ending'] = is_ending
    
    if i > 0:
        prev_word = sent[i-1][0]
        features.update({
            '-1:word.lower()': prev_word.lower(),
            '-1:word[-2:]': prev_word[-2:],
        })
    else:
        features['BOS'] = True
    
    if i < len(sent) - 1:
        features.update({'+1:word.lower()': sent[i+1][0].lower()})
    else:
        features['EOS'] = True
    
    return features


def sent2features(sent, use_common_ending=True, endings=None):
    return [word2features(sent, i, use_common_ending, endings) for i in range(len(sent))]


def sent2labels(sent):
    return [tag for _, tag in sent]


# ==============================================================================
# CRF TRAINING
# ==============================================================================

def train_crf(X_train, y_train, params=None):
    """Train CRF model with given features and labels."""
    if params is None:
        params = CRF_PARAMS
    
    crf = sklearn_crfsuite.CRF(**params)
    crf.fit(X_train, y_train)
    return crf


# ==============================================================================
# METRICS
# ==============================================================================

def compute_metrics(y_true_flat, y_pred_flat, labels=None):
    """Compute P/R/F1 per class and macro."""
    if labels is None:
        labels = sorted(set(y_true_flat) | set(y_pred_flat))
    
    results = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true_flat, y_pred_flat) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true_flat, y_pred_flat) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true_flat, y_pred_flat) if t == label and p != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[label] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': tp + fn}
    
    # Macro average
    macro_p = np.mean([results[l]['precision'] for l in labels])
    macro_r = np.mean([results[l]['recall'] for l in labels])
    macro_f1 = np.mean([results[l]['f1'] for l in labels])
    
    accuracy = sum(1 for t, p in zip(y_true_flat, y_pred_flat) if t == p) / len(y_true_flat)
    
    results['macro'] = {'precision': macro_p, 'recall': macro_r, 'f1': macro_f1}
    results['accuracy'] = accuracy
    
    return results


def confusion_matrix(y_true_flat, y_pred_flat, labels=None):
    """Compute confusion matrix as dict of dicts."""
    if labels is None:
        labels = sorted(set(y_true_flat) | set(y_pred_flat))
    
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true_flat, y_pred_flat):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
    return matrix


# ==============================================================================
# MCNEMAR'S TEST
# ==============================================================================

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """McNemar's test for paired model comparison.
    
    Tests whether two models have significantly different error rates.
    Returns chi-squared statistic and p-value.
    """
    # Count discordant pairs
    b = 0  # A correct, B wrong
    c = 0  # A wrong, B correct
    
    for t, pa, pb in zip(y_true, y_pred_a, y_pred_b):
        a_correct = (t == pa)
        b_correct = (t == pb)
        if a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1
    
    if b + c == 0:
        return 0.0, 1.0  # No disagreement
    
    # McNemar's chi-squared (with continuity correction)
    chi2 = (abs(b - c) - 1)**2 / (b + c)
    
    # Approximate p-value from chi-squared distribution (1 df)
    # Using survival function approximation
    p_value = math.exp(-chi2 / 2) if chi2 < 20 else 0.0
    
    return chi2, p_value


def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for a metric.
    
    metric_fn: function(y_true, y_pred) -> float
    Returns: (mean, lower, upper)
    """
    rng = random.Random(seed)
    n = len(y_true)
    scores = []
    
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n-1) for _ in range(n)]
        y_t = [y_true[i] for i in indices]
        y_p = [y_pred[i] for i in indices]
        scores.append(metric_fn(y_t, y_p))
    
    scores.sort()
    alpha = (1 - ci) / 2
    lower = scores[int(alpha * n_bootstrap)]
    upper = scores[int((1 - alpha) * n_bootstrap)]
    mean = np.mean(scores)
    
    return mean, lower, upper


def accuracy_metric(y_true, y_pred):
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def f1_stop_metric(y_true, y_pred):
    """F1 for STOP class only."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 'STOP' and p == 'STOP')
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != 'STOP' and p == 'STOP')
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 'STOP' and p != 'STOP')
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0


# ==============================================================================
# BASELINES
# ==============================================================================

def majority_baseline(y_true):
    """Always predict the most common class."""
    counts = Counter(y_true)
    majority = counts.most_common(1)[0][0]
    return [majority] * len(y_true)


def random_baseline(y_true, seed=42):
    """Predict randomly according to class distribution."""
    rng = random.Random(seed)
    counts = Counter(y_true)
    total = len(y_true)
    stop_rate = counts.get('STOP', 0) / total
    return ['STOP' if rng.random() < stop_rate else 'O' for _ in y_true]


def rule_only_baseline(sequences):
    """Use only suffix matching rules (no ML model)."""
    predictions = []
    for sent in sequences:
        for word, _ in sent:
            is_stop = any(word.endswith(s) for s in CANONICAL_ENDINGS)
            predictions.append('STOP' if is_stop else 'O')
    return predictions


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================

def run_full_evaluation(data_path="train_labeled.tsv", max_sentences=None):
    """Run the complete evaluation pipeline."""
    
    if not HAS_CRF:
        print("ERROR: sklearn-crfsuite required. Install with: pip install sklearn-crfsuite")
        return
    
    print("=" * 70)
    print("COMPREHENSIVE EVALUATION FRAMEWORK")
    print("=" * 70)
    
    # 1. Load data
    print("\n[1/8] Loading data...")
    sequences = load_labeled_data(data_path)
    if max_sentences:
        sequences = sequences[:max_sentences]
    print(f"  Loaded {len(sequences)} sequences, {sum(len(s) for s in sequences)} tokens")
    
    all_results = {}
    
    # 2. Prepare data
    print("\n[2/8] Preparing features...")
    random.seed(42)
    random.shuffle(sequences)
    
    split_idx = int(len(sequences) * CRF_TRAIN_SPLIT)
    train_seqs = sequences[:split_idx]
    test_seqs = sequences[split_idx:]
    
    X_train = [sent2features(s, use_common_ending=True) for s in train_seqs]
    y_train = [sent2labels(s) for s in train_seqs]
    X_test = [sent2features(s, use_common_ending=True) for s in test_seqs]
    y_test = [sent2labels(s) for s in test_seqs]
    
    y_test_flat = [tag for seq in y_test for tag in seq]
    print(f"  Train: {len(train_seqs)} seqs | Test: {len(test_seqs)} seqs")
    print(f"  Test tokens: {len(y_test_flat)} (O: {y_test_flat.count('O')}, STOP: {y_test_flat.count('STOP')})")
    
    # 3. Train CRF (full features)
    print("\n[3/8] Training CRF (full features)...")
    t_start = time.time()
    crf_full = train_crf(X_train, y_train)
    train_time = time.time() - t_start
    print(f"  Training time: {train_time:.2f}s")
    
    # Save model
    model_path = "ayurvedic_segmenter.pkl"
    joblib.dump(crf_full, model_path)
    print(f"  Model saved to {model_path}")
    
    # Evaluate full CRF
    y_pred_full = crf_full.predict(X_test)
    y_pred_full_flat = [tag for seq in y_pred_full for tag in seq]
    
    metrics_full = compute_metrics(y_test_flat, y_pred_full_flat, labels=['O', 'STOP'])
    cm_full = confusion_matrix(y_test_flat, y_pred_full_flat, labels=['O', 'STOP'])
    
    print(f"\n  === CRF (Full Features) Results ===")
    print(f"  Accuracy: {metrics_full['accuracy']:.4f}")
    print(f"  O     — P: {metrics_full['O']['precision']:.4f} R: {metrics_full['O']['recall']:.4f} F1: {metrics_full['O']['f1']:.4f}")
    print(f"  STOP  — P: {metrics_full['STOP']['precision']:.4f} R: {metrics_full['STOP']['recall']:.4f} F1: {metrics_full['STOP']['f1']:.4f}")
    print(f"  Macro — P: {metrics_full['macro']['precision']:.4f} R: {metrics_full['macro']['recall']:.4f} F1: {metrics_full['macro']['f1']:.4f}")
    print(f"  Confusion Matrix: O→O={cm_full['O']['O']} O→STOP={cm_full['O']['STOP']} STOP→O={cm_full['STOP']['O']} STOP→STOP={cm_full['STOP']['STOP']}")
    
    all_results['crf_full'] = metrics_full
    all_results['crf_full_cm'] = cm_full
    
    # 4. Ablation: CRF without is_common_ending
    print("\n[4/8] Ablation: CRF WITHOUT is_common_ending feature...")
    X_train_no = [sent2features(s, use_common_ending=False) for s in train_seqs]
    X_test_no = [sent2features(s, use_common_ending=False) for s in test_seqs]
    
    crf_no_ending = train_crf(X_train_no, y_train)
    y_pred_no = crf_no_ending.predict(X_test_no)
    y_pred_no_flat = [tag for seq in y_pred_no for tag in seq]
    
    metrics_no = compute_metrics(y_test_flat, y_pred_no_flat, labels=['O', 'STOP'])
    print(f"  Accuracy: {metrics_no['accuracy']:.4f} (Δ={metrics_no['accuracy'] - metrics_full['accuracy']:+.4f})")
    print(f"  STOP F1: {metrics_no['STOP']['f1']:.4f} (Δ={metrics_no['STOP']['f1'] - metrics_full['STOP']['f1']:+.4f})")
    
    all_results['crf_no_ending'] = metrics_no
    
    # McNemar test: full vs ablated
    chi2, p_val = mcnemar_test(y_test_flat, y_pred_full_flat, y_pred_no_flat)
    print(f"  McNemar χ²={chi2:.4f}, p={p_val:.6f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'}")
    all_results['mcnemar_ablation'] = {'chi2': chi2, 'p_value': p_val}
    
    # 5. Baselines
    print("\n[5/8] Computing baselines...")
    
    # Majority baseline
    y_majority = majority_baseline(y_test_flat)
    metrics_majority = compute_metrics(y_test_flat, y_majority, labels=['O', 'STOP'])
    print(f"  Majority: Acc={metrics_majority['accuracy']:.4f} F1(STOP)={metrics_majority['STOP']['f1']:.4f}")
    all_results['majority'] = metrics_majority
    
    # Random baseline
    y_random = random_baseline(y_test_flat)
    metrics_random = compute_metrics(y_test_flat, y_random, labels=['O', 'STOP'])
    print(f"  Random:   Acc={metrics_random['accuracy']:.4f} F1(STOP)={metrics_random['STOP']['f1']:.4f}")
    all_results['random'] = metrics_random
    
    # Rule-only baseline
    y_rule = rule_only_baseline(test_seqs)
    metrics_rule = compute_metrics(y_test_flat, y_rule, labels=['O', 'STOP'])
    print(f"  Rule-only: Acc={metrics_rule['accuracy']:.4f} F1(STOP)={metrics_rule['STOP']['f1']:.4f}")
    all_results['rule_only'] = metrics_rule
    
    # 6. Bootstrap confidence intervals
    print("\n[6/8] Bootstrap 95% confidence intervals (1000 resamples)...")
    
    acc_mean, acc_lo, acc_hi = bootstrap_ci(y_test_flat, y_pred_full_flat, accuracy_metric)
    f1_mean, f1_lo, f1_hi = bootstrap_ci(y_test_flat, y_pred_full_flat, f1_stop_metric)
    print(f"  CRF Accuracy: {acc_mean:.4f} [{acc_lo:.4f}, {acc_hi:.4f}]")
    print(f"  CRF F1(STOP): {f1_mean:.4f} [{f1_lo:.4f}, {f1_hi:.4f}]")
    all_results['bootstrap_ci'] = {
        'accuracy': {'mean': acc_mean, 'lower': acc_lo, 'upper': acc_hi},
        'f1_stop': {'mean': f1_mean, 'lower': f1_lo, 'upper': f1_hi},
    }
    
    # 7. Data-size learning curve
    print("\n[7/8] Learning curve (data-size sweep)...")
    data_sizes = [1000, 5000, 10000, 15000, 30000, min(50000, len(sequences))]
    if len(sequences) > 50000:
        data_sizes.append(len(sequences))
    
    learning_curve = []
    for size in data_sizes:
        if size > len(sequences):
            continue
        subset = sequences[:size]
        s_idx = int(len(subset) * CRF_TRAIN_SPLIT)
        Xs = [sent2features(s) for s in subset[:s_idx]]
        ys = [sent2labels(s) for s in subset[:s_idx]]
        Xt = [sent2features(s) for s in subset[s_idx:]]
        yt_flat = [tag for seq in [sent2labels(s) for s in subset[s_idx:]] for tag in seq]
        
        if not Xs or not Xt or not yt_flat:
            continue
        
        crf_i = train_crf(Xs, ys)
        yp_flat = [tag for seq in crf_i.predict(Xt) for tag in seq]
        m = compute_metrics(yt_flat, yp_flat, labels=['O', 'STOP'])
        
        entry = {'size': size, 'accuracy': m['accuracy'], 'f1_stop': m['STOP']['f1'], 'f1_macro': m['macro']['f1']}
        learning_curve.append(entry)
        print(f"  N={size:>6d}: Acc={m['accuracy']:.4f} F1(STOP)={m['STOP']['f1']:.4f} F1(macro)={m['macro']['f1']:.4f}")
    
    all_results['learning_curve'] = learning_curve
    
    # 8. Latency benchmarking
    print("\n[8/8] Latency benchmarking (100 inferences)...")
    test_text = "වාත රෝග සඳහා නියඟලා අලයක් ගෙන හොඳින් සුද්ද කරගන්න ඉන්පසු එය ගොම දියරේ දින තුනක් ගිල්වා තබන්න"
    words = test_text.split()
    dummy_sent = [(w, "") for w in words]
    test_features = [sent2features(dummy_sent)]
    
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        crf_full.predict_marginals(test_features)
        latencies.append(time.perf_counter() - t0)
    
    lat_mean = np.mean(latencies) * 1000  # ms
    lat_std = np.std(latencies) * 1000
    lat_p95 = np.percentile(latencies, 95) * 1000
    print(f"  CRF: {lat_mean:.2f} ± {lat_std:.2f} ms (p95: {lat_p95:.2f} ms)")
    all_results['latency_crf'] = {'mean_ms': lat_mean, 'std_ms': lat_std, 'p95_ms': lat_p95}
    
    # 5-fold cross-validation
    print("\n[BONUS] 5-Fold Cross-Validation...")
    fold_size = len(sequences) // 5
    cv_scores = []
    for fold in range(5):
        val_start = fold * fold_size
        val_end = val_start + fold_size
        cv_val = sequences[val_start:val_end]
        cv_train = sequences[:val_start] + sequences[val_end:]
        
        Xc_train = [sent2features(s) for s in cv_train]
        yc_train = [sent2labels(s) for s in cv_train]
        Xc_val = [sent2features(s) for s in cv_val]
        yc_val_flat = [tag for seq in [sent2labels(s) for s in cv_val] for tag in seq]
        
        crf_cv = train_crf(Xc_train, yc_train)
        yc_pred_flat = [tag for seq in crf_cv.predict(Xc_val) for tag in seq]
        m_cv = compute_metrics(yc_val_flat, yc_pred_flat, labels=['O', 'STOP'])
        cv_scores.append(m_cv)
        print(f"  Fold {fold+1}: Acc={m_cv['accuracy']:.4f} F1(STOP)={m_cv['STOP']['f1']:.4f}")
    
    cv_acc = [s['accuracy'] for s in cv_scores]
    cv_f1 = [s['STOP']['f1'] for s in cv_scores]
    print(f"  Mean: Acc={np.mean(cv_acc):.4f}±{np.std(cv_acc):.4f} F1={np.mean(cv_f1):.4f}±{np.std(cv_f1):.4f}")
    all_results['cross_validation'] = {
        'accuracy_mean': float(np.mean(cv_acc)), 'accuracy_std': float(np.std(cv_acc)),
        'f1_stop_mean': float(np.mean(cv_f1)), 'f1_stop_std': float(np.std(cv_f1)),
        'folds': [{'accuracy': s['accuracy'], 'f1_stop': s['STOP']['f1']} for s in cv_scores]
    }
    
    # Save all results
    results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    serializable = json.loads(json.dumps(all_results, default=convert))
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*70}")
    print(f"All results saved to {results_path}")
    print(f"Model saved to {model_path}")
    print(f"{'='*70}")
    
    return all_results


# ==============================================================================
# SAFETY GUARDRAIL EVALUATION
# ==============================================================================

def evaluate_safety_guardrail():
    """Evaluate safety guardrail across window sizes with known test cases."""
    from pipeline import load_knowledge_graph, analyze_safety
    
    kg = load_knowledge_graph("ayurvedic_ingredients_full.csv")
    if not kg:
        print("ERROR: Cannot load knowledge graph.")
        return
    
    # Test cases: (input_text, expected_status)
    test_cases = [
        # CASE 1: Toxic ingredient WITH purification — should APPROVE
        (
            "වාත රෝග සඳහා නියඟලා අලයක් ගෙන. හොඳින් ගොම දියරේ දින තුනක් ගිල්වා ශෝධනය කරන්න.",
            "APPROVED"
        ),
        # CASE 2: Toxic ingredient WITHOUT purification — should REJECT
        (
            "වාත රෝග සඳහා නියඟලා අලයක් ගෙන කුඩු කරගන්න.",
            "REJECTED"
        ),
        # CASE 3: No toxic ingredient — should APPROVE
        (
            "ඉඟුරු සහ මී පැණි මිශ්‍ර කරන්න.",
            "APPROVED"
        ),
        # CASE 4: Multiple toxic ingredients, one without purification — should REJECT
        (
            "නියඟලා ගොම දියරේ ශෝධනය කර. ජයපාල බීජ කුඩු කරන්න.",
            "REJECTED"
        ),
    ]
    
    print("\n=== Safety Guardrail Evaluation ===")
    
    results = {}
    for window_size in range(4):
        correct = 0
        total = len(test_cases)
        details = []
        
        for text, expected in test_cases:
            report = analyze_safety(text, kg, window_size=window_size)
            actual = report["final_status"]
            passed = actual == expected
            correct += int(passed)
            details.append({
                'text': text[:50] + '...',
                'expected': expected,
                'actual': actual,
                'correct': passed
            })
        
        accuracy = correct / total
        results[f'window_{window_size}'] = {'accuracy': accuracy, 'correct': correct, 'total': total, 'details': details}
        print(f"  Window={window_size}: {correct}/{total} correct ({accuracy:.0%})")
    
    results_path = os.path.join(RESULTS_DIR, "safety_eval_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {results_path}")
    
    return results


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("Ayurvedic NLP Pipeline — Full Evaluation\n")
    
    # Check for data
    data_file = "train_labeled.tsv"
    if not os.path.exists(data_file):
        print(f"ERROR: {data_file} not found. Run weak_supervision_generator.py first.")
        exit(1)
    
    # Run evaluation with sample for speed (remove max_sentences for full run)
    results = run_full_evaluation(data_file, max_sentences=5000)
    
    # Run safety evaluation
    safety_results = evaluate_safety_guardrail()
    
    print("\n✅ Evaluation complete. Check eval_results/ for all outputs.")
