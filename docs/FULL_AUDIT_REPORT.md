# COMPREHENSIVE RESEARCH AUDIT REPORT

## Neuro-Symbolic NLP Pipeline for Archaic Sinhala Ayurvedic Manuscript Processing

### Full-Scale End-to-End Analysis

**Audit Date:** April 2026  
**Scope:** Code quality, methodology, data integrity, edge cases, gaps, and thesis readiness

---

## EXECUTIVE SUMMARY

This project implements a multi-phase NLP pipeline for processing ancient Sinhala Ayurvedic palm-leaf manuscripts, combining OCR post-correction, sentence boundary detection, and toxic ingredient safety validation. The research makes a valid and novel contribution at the intersection of **low-resource NLP**, **historical document processing**, and **safety-critical medical AI**.

**Overall Assessment:** The project is approximately **65-70% thesis-ready**. The conceptual architecture is strong, but there are significant gaps in formal evaluation, reproducibility, ablation studies, missing model artifacts, and statistical rigor that must be addressed before PhD-level defense.

---

## TABLE OF CONTENTS

1. [Architecture Audit](#1-architecture-audit)
2. [Data Quality Audit](#2-data-quality-audit)
3. [Code Quality Audit](#3-code-quality-audit)
4. [Methodology Audit](#4-methodology-audit)
5. [Edge Cases & Failure Modes](#5-edge-cases--failure-modes)
6. [Missing Components (Critical)](#6-missing-components-critical)
7. [Missing Components (Important)](#7-missing-components-important)
8. [Statistical Rigor Gaps](#8-statistical-rigor-gaps)
9. [Literature & Related Work Gaps](#9-literature--related-work-gaps)
10. [Thesis Structural Requirements](#10-thesis-structural-requirements)
11. [Research Paper Requirements](#11-research-paper-requirements)
12. [Demo Implementation Gaps](#12-demo-implementation-gaps)
13. [Actionable Recommendations (Priority-Ranked)](#13-actionable-recommendations)

---

## 1. ARCHITECTURE AUDIT

### 1.1 Pipeline Architecture: ✅ SOUND

```
OCR Scanner → Bigram LM + Viterbi (Phase 1A) → CRF/RoBERTa Segmenter (Phase 1B/2) → KG Safety Guardrail (Phase 1C)
```

**Strengths:**

- Well-designed 3-stage pipeline with clear separation of concerns
- Neuro-symbolic hybrid approach (statistical + neural + deterministic) is architecturally novel
- Hard-veto safety guardrail is the correct design for medical AI — never overridden by soft scores
- Human-in-the-loop context window slider bridges the safety-vs-context trade-off elegantly

**Weaknesses:**

- ❌ No end-to-end integration test exists — each stage is tested in isolation but never as a full pipeline
- ❌ No error propagation analysis — how does OCR error rate affect downstream segmentation accuracy, and how does that compound into safety guardrail false positive/negative rates?
- ❌ The Viterbi decoder operates on simulated OCR JSON data, not actual OCR output — the pipeline has never been tested on a real palm-leaf scan
- ⚠️ The CRF inference in `pipeline.py` uses a different suffix list than the CRF training in the Phase 1 notebook (the notebook uses 9 suffixes; `pipeline.py` uses 13). This feature mismatch could cause inference-time errors.

### 1.2 Feature Mismatch Bug (CRITICAL)

**In Phase 1 Notebook (Training):**

```python
common_endings = ['යි', 'ස්', 'යුතු', 'යේය', 'වේ', 'මැනවි', 'ගනු', 'පෙර', 'පසු']  # 9 items
```

**In pipeline.py (Inference/Production):**

```python
common_endings = ['යි', 'ස්', 'යුතු', 'යේය', 'වේ', 'මැනවි', 'ගනු', 'පෙර', 'පසු', 'කරයි', 'න්න', 'ගන්න', 'තබන්න']  # 13 items
```

**Impact:** The CRF model was trained on features computed with 9 suffixes but is being asked to predict using features computed with 13 suffixes. Feature `is_common_ending` will fire `True` for different words at inference time than it did during training. This could degrade segmentation accuracy unpredictably. **This must be fixed immediately.**

---

## 2. DATA QUALITY AUDIT

### 2.1 Dataset Statistics

| File                               | Lines                     | Status       |
| ---------------------------------- | ------------------------- | ------------ |
| `train.txt`                        | 70,000                    | ✅ Present   |
| `train_labeled.tsv`                | 852,140                   | ✅ Present   |
| `cleaned_corpus.txt` (Phase 1 & 2) | 70,000 each               | ✅ Present   |
| `ayurvedic_ingredients_full.csv`   | 2,101 rows (incl. header) | ✅ Present   |
| `ayurvedic_ingredients_sample.csv` | ~10 rows                  | ✅ Present   |
| `bigram_probabilities.json`        | Present                   | ✅ Generated |
| `ayurvedic_segmenter.pkl`          | **NOT FOUND**             | ❌ MISSING   |

### 2.2 Critical Data Issues

#### ❌ Missing Trained Model File

The file `ayurvedic_segmenter.pkl` (the trained CRF model) is **not present** in the repository. The Streamlit app (`app.py`) and `pipeline.py` both depend on it. Without this file, the demo cannot run.

#### ⚠️ Weak Supervision Circularity

The `weak_supervision_generator.py` creates `train_labeled.tsv` by applying the **same morphological rules** that are later used as CRF features. This creates a circularity:

- Training data is generated using suffix rules
- CRF is trained with those same suffix rules as features
- The CRF is essentially learning to replicate the weak supervision heuristic

**Thesis Risk:** A PhD examiner will immediately question: "How do you know the CRF is learning anything beyond the weak supervision rules?" You need an ablation study showing the CRF learns **contextual patterns beyond** the injected features.

#### ⚠️ Forced Last-Word STOP Tags

In `weak_supervision_generator.py`:

```python
if i == len(words) - 1:
    is_boundary = True
```

Every last word of every line is forced to be `STOP`. This means **all training labels at sentence ends are STOP regardless of the actual suffix**. The CRF may learn that "being the last word" = STOP, which is a positional artifact rather than linguistic understanding. This inflates the apparent accuracy.

#### ⚠️ Knowledge Graph Quality

- The CSV has 2,100 entries, but many are **systematic expansions** (நியങ්லா + 8 plant-part variants = 9 rows for one entity). The actual unique entity count is much lower (~50-60 core entities with ~8-9 part variants each).
- **No source citations** for toxicity levels or purification keywords. For a medical safety system, this is unacceptable — every toxicity classification must be traceable to an authoritative Ayurvedic pharmacopoeia (e.g., Charaka Samhita, Sushruta Samhita, or Sri Lankan Pharmacopoeia).
- **No dosage-dependent toxicity.** Many "Low" toxicity herbs become toxic at high doses (e.g., nutmeg/සාදික්කා). The binary High/Medium/Low classification is oversimplified.

#### ⚠️ Dataset Provenance Unknown

- Where did the 70,000 sentences come from? OCR of actual manuscripts? Manual transcription? Generated? This provenance is never documented.
- What manuscripts specifically? Palm-leaf collections from which library/museum?
- What century/era do the texts represent?
- Is there any copyright or institutional approval required?

---

## 3. CODE QUALITY AUDIT

### 3.1 pipeline.py

| Aspect           | Status      | Notes                                                                                              |
| ---------------- | ----------- | -------------------------------------------------------------------------------------------------- |
| Error handling   | ⚠️ Minimal  | `FileNotFoundError` only; no handling of malformed CSV, corrupt model files                        |
| Input validation | ❌ Missing  | No check for maximum input length, special characters, or encoding issues                          |
| Security         | ⚠️ Risk     | CSV loading with no path sanitization; potential for path traversal if CSV path is user-controlled |
| Performance      | ✅ Adequate | Simple loops, reasonable for this scale                                                            |
| Test coverage    | ❌ None     | No unit tests exist for any function                                                               |

### 3.2 viterbi_decoder.py

| Aspect           | Status     | Notes                                                                                                                                                      |
| ---------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Correctness      | ⚠️ Issue   | Smoothing constant `0.0001` is hardcoded — should be a parameter; also no proper Laplace/Add-k smoothing implemented despite being claimed in the document |
| Score arithmetic | ⚠️ Issue   | Multiplicative scoring `V[t-1] * (alpha*OCR + beta*LM)` means scores shrink rapidly toward zero for longer sequences. Log-space computation should be used |
| Edge cases       | ❌ Issue   | If `ocr_input` has only one position, the backtracking loop will fail                                                                                      |
| OOV handling     | ❌ Missing | No handling for OOV words in language model — claimed but not implemented                                                                                  |

### 3.3 testing.py

This actually implements bigram model TRAINING, not testing. Misleading filename.

### 3.4 app.py (Streamlit)

| Aspect         | Status     | Notes                                              |
| -------------- | ---------- | -------------------------------------------------- |
| UI design      | ✅ Good    | Bilingual labels (Sinhala + English), clear layout |
| Error handling | ⚠️ Partial | Checks for missing KG but not missing CRF model    |
| Responsiveness | ✅ Good    | Uses spinners and caching appropriately            |
| Accessibility  | ⚠️ Partial | Emoji-based status indicators; no ARIA labels      |

### 3.5 weak_supervision_generator.py

| Aspect               | Status   | Notes                                                                                             |
| -------------------- | -------- | ------------------------------------------------------------------------------------------------- |
| Correctness          | ⚠️ Issue | Path uses `../train.txt` (relative parent) — fragile                                              |
| Suffix list mismatch | ⚠️ Issue | Contains 16 suffixes vs. 9 in notebook training vs. 13 in pipeline.py — THREE different versions! |
| Reproducibility      | ✅ OK    | Deterministic (no randomness)                                                                     |

---

## 4. METHODOLOGY AUDIT

### 4.1 Experimental Design: ⚠️ WEAK

**What's claimed but not proven:**

| Claim                                      | Evidence Provided                  | Evidence Missing                                                        |
| ------------------------------------------ | ---------------------------------- | ----------------------------------------------------------------------- |
| "CRF outperforms RoBERTa at 15K sentences" | Stated in text                     | ❌ No actual F1/Precision/Recall numbers for either model at 15K        |
| "RoBERTa achieves 92% accuracy at 70K"     | Stated in thesis_research_notes.md | ❌ No training logs, no confusion matrix, no per-class metrics          |
| "Bigram + Viterbi achieves 75% accuracy"   | Stated in text                     | ❌ No evaluation dataset, no ground truth OCR pairs, no WER/CER metrics |
| "CRF achieves 0.05s latency"               | Stated in text                     | ❌ No latency benchmarking code or log                                  |
| "RoBERTa achieves 1.2s latency"            | Stated in text                     | ❌ No latency benchmarking code or log                                  |

**Critical Gap:** There are **ZERO formal evaluation results** anywhere in the codebase. No saved metrics, no evaluation scripts, no confusion matrices, no classification reports. All performance claims are stated without evidence. This is the single largest gap for thesis defense.

### 4.2 Missing Ablation Studies

For PhD-level defense, the following ablation experiments are required:

1. **CRF without `is_common_ending` feature** — Does the model still work? What's the F1 drop? This proves the feature's value.
2. **CRF with only `is_common_ending`** — Is this feature alone sufficient? (If yes, you don't need a CRF at all.)
3. **RoBERTa at {5K, 10K, 15K, 30K, 50K, 70K} data sizes** — Proper learning curve showing the data-to-performance relationship.
4. **CRF at the same data sizes** — You must compare on equal footing.
5. **Window size {0, 1, 2, 3} safety analysis** — How do false positive/false negative rates change?
6. **Viterbi alpha/beta sweep** — Performance at different OCR trust vs. LM trust ratios.

### 4.3 Missing Baselines

No comparison against obvious baselines:

- **Punctuation heuristic only** (just check for the suffix rules without any ML model)
- **Simple rule-based segmenter** (if-then rules on verb endings)
- **Random baseline** (randomly place STOP tags at the empirical STOP rate)
- **Majority class baseline** (always predict O)
- **Regular expression baseline** (regex matching sentence-end patterns)

Without these baselines, you cannot prove the CRF/RoBERTa adds value beyond simple rules.

---

## 5. EDGE CASES & FAILURE MODES

### 5.1 Segmentation Edge Cases NOT Handled

1. **Compound sentences with multiple clauses:** "නියඟලා ගෙන සුද්ද කරගනු මැනවි ඉන්පසු ගොම දියරේ ගිල්වා තබනු මැනවි" — Two instructions in one line with "මැනවි" appearing twice. First "මැනවි" is a clause boundary, not a sentence boundary.

2. **Quoted speech in manuscripts:** Ancient texts sometimes quote previous scholars. Quoted speech ending in verb suffixes should not trigger sentence boundaries.

3. **Enumerations/lists:** "ඉඟුරු, කුරුඳු, ගම්මිරිස් යොදනු මැනවි" — A list followed by a single verb. Each item should NOT be treated as a sentence.

4. **Copula omission:** In archaic Sinhala, the copula (be-verb) is often dropped. Some sentences end without any verb whatsoever.

5. **Sanskrit borrowings:** Many Ayurvedic terms are Sanskrit-origin and may contain suffixes that look like Sinhala verb endings but aren't.

### 5.2 Safety Guardrail Edge Cases NOT Handled

1. **Partial name matching failure:** Entity "නියඟලා" won't match "නියංගලා" (common spelling variation). No fuzzy matching or Unicode normalization is implemented.

2. **Negation blindness:** "නියඟලා යොදා ගැනීම නොකළ යුතුය" (Niyangala should NOT be used) — The guardrail would flag this for missing purification, but the text is actually REJECTING the ingredient. The system cannot understand negation.

3. **Conditional instructions:** "රෝගය වැඩි නම් පමණක් නියඟලා යොදන්න" (Use Niyangala ONLY IF the disease is severe) — Conditional usage context is ignored.

4. **Dosage-dependent toxicity:** "නියඟලා තුනි පමණක්" (Only a tiny amount of Niyangala) — Microdoses may be safe without full purification. The system has no concept of dosage.

5. **Multi-step purification missing partial steps:** If purification requires 3 steps and only 2 are mentioned, the current system would PASS because it only checks for keyword presence, not completeness.

6. **Temporal ordering blindness:** "කුඩු කර ගොම දියරේ බහන්න" (Powder it THEN soak in cow urine) vs. "ගොම දියරේ බහා කුඩු කරන්න" (Soak THEN powder). The order matters pharmacologically but the guardrail only checks keyword presence.

7. **Cross-recipe contamination with window > 0:** With `window_size=1`, a purification keyword from an adjacent DIFFERENT recipe could falsely validate a toxic recipe.

### 5.3 OCR Correction Edge Cases NOT Handled

1. **Candidate list ordering:** If the correct word is not in the candidate list at all, the Viterbi decoder cannot produce it (closed vocabulary).
2. **Homographs:** Different words with identical spelling but different meanings in context.
3. **Score underflow:** Multiplicative scoring without log-space will underflow to 0.0 for sequences longer than ~20 words.
4. **Empty candidate lists:** No handling for positions with zero candidates.

---

## 6. MISSING COMPONENTS (CRITICAL — Must Fix for Thesis)

### 6.1 ❌ Formal Evaluation Framework

**What's needed:** A complete evaluation script that:

- Loads a held-out test set with **ground-truth segment boundaries**
- Runs the CRF segmenter and measures Precision, Recall, F1 (per-class and macro)
- Runs the RoBERTa segmenter on the same data
- Computes statistical significance tests (McNemar's test or paired bootstrap)
- Generates confusion matrices
- Saves all results to files for reproducibility

### 6.2 ❌ Gold Standard Test Set

**What's needed:** A manually-annotated test set of at least 200-500 sentences where a domain expert (Ayurvedic practitioner or Sinhala linguist) has placed correct sentence boundaries. This is what you evaluate against. Evaluating against weak supervision labels is circular.

### 6.3 ❌ Trained Model Artifact (ayurvedic_segmenter.pkl)

The trained CRF model `.pkl` file is missing. It must be either:

- Committed to the repository, or
- Reproducibly generated by running the Phase 1 notebook

### 6.4 ❌ End-to-End Integration Test

A single script that takes raw text as input, runs it through all 3 stages (OCR correction → segmentation → safety), and produces the final safety report. Currently each stage exists in isolation.

### 6.5 ❌ Ablation Study Results

At minimum: CRF with and without `is_common_ending` feature, and RoBERTa learning curve across data sizes.

### 6.6 ❌ Error Propagation Analysis

Quantify: "If Stage 1A has X% WER, how does that translate to Y% drop in Stage 1B segmentation accuracy, and Z% change in Stage 1C false positive/negative rates?"

---

## 7. MISSING COMPONENTS (IMPORTANT — Should Fix)

### 7.1 ⚠️ Cross-Validation

The 80/20 single split is insufficient for reliable results. Use 5-fold or 10-fold cross-validation and report mean ± std for all metrics.

### 7.2 ⚠️ Hyperparameter Tuning Documentation

The CRF uses c1=0.1, c2=0.1 without justification. Was a grid search performed? What about the segmentation threshold (0.15)?

### 7.3 ⚠️ Latency Benchmarking

Implement a script that runs inference N times and reports mean ± std latency for both CRF and RoBERTa.

### 7.4 ⚠️ Unicode Normalization

Sinhala Unicode has multiple ways to represent the same character (NFC vs NFD normalization). No normalization is applied to any input. This could cause entity matching failures.

### 7.5 ⚠️ Logging and Experiment Tracking

No experiment tracking (MLflow, W&B, or even simple CSV logging). Training runs are not reproducible.

### 7.6 ⚠️ Confidence Calibration

The CRF `predict_marginals` probabilities should be calibrated. Are they reliable? A reliability diagram should be plotted.

---

## 8. STATISTICAL RIGOR GAPS

### 8.1 No Significance Testing

Claiming "CRF outperforms RoBERTa" without a statistical significance test is not publishable. Required:

- McNemar's test (for paired predictions)
- Paired bootstrap confidence intervals
- p-value < 0.05 with Bonferroni correction for multiple comparisons

### 8.2 No Confidence Intervals

All metrics should be reported as mean ± 95% CI. A single-run F1 score is meaningless without variance estimates.

### 8.3 No Inter-Annotator Agreement

If the gold standard is created by a single annotator, there's no way to measure label reliability. Even for weak supervision, you should have a small manually-annotated subset with IAA (Cohen's kappa > 0.8).

### 8.4 No Effect Size Reporting

Beyond p-values, report Cohen's d or similar effect sizes to demonstrate practical significance.

---

## 9. LITERATURE & RELATED WORK GAPS

### 9.1 References Mentioned but Not Formally Cited

- "Rijhwani et al. (EMNLP 2020)" — mentioned without proper citation
- "Ruokolainen et al. (ACL)" — year missing
- "CHIPMUNK" — no proper reference

### 9.2 Missing Related Work Categories

| Topic                               | What You Need to Cite                                                                        |
| ----------------------------------- | -------------------------------------------------------------------------------------------- |
| Sinhala NLP                         | De Silva & Weerasinghe (2010), Fernando et al. SinLing corpus, UCL Sinhala NLP projects      |
| Low-resource NLP                    | Joshi et al. (2020) "State and Fate of Linguistic Diversity", Hedderich et al. (2021) Survey |
| Weak supervision                    | Ratner et al. "Snorkel" (2017), Lison et al. for NER (2020)                                  |
| Sentence boundary detection         | Read et al. (2012), Schweter & Ahmed (2019)                                                  |
| OCR post-correction                 | Nguyen et al. (2020), Rigaud et al. (2019)                                                   |
| CRF for sequence labeling           | Lafferty et al. (2001) — the foundational CRF paper                                          |
| Medical NLP safety                  | Johnson et al. "MIMIC", Lehman et al. safety guardrails                                      |
| Ayurvedic toxicology                | Bhaishajya Ratnavali, Rasa Tarangini (classical references)                                  |
| Knowledge graphs for medical safety | Rotmensch et al. (2017), Zhang et al. medical KGs                                            |
| Human-in-the-loop ML                | Monarch (2021), Wu et al. HITL for NLP                                                       |
| XLM-RoBERTa                         | Conneau et al. (2020) — the original paper                                                   |
| Viterbi algorithm                   | Viterbi (1967) original, Forney (1973)                                                       |

### 9.3 Missing Theoretical Framework

No formal framework connecting:

- Information Theory (Shannon's noisy channel → OCR correction)
- Graphical Models (CRF theory → segmentation)
- Transfer Learning (multilingual → low-resource fine-tuning)
- Knowledge Representation (KG formalism → safety validation)

---

## 10. THESIS STRUCTURAL REQUIREMENTS

### For a PhD-level thesis, the following chapters are required:

| Chapter                                       | Status              | Notes                                                             |
| --------------------------------------------- | ------------------- | ----------------------------------------------------------------- |
| 1. Introduction                               | ❌ Not written      | Problem statement, motivation, research questions, contributions  |
| 2. Literature Review                          | ❌ Not written      | Comprehensive survey of related work (see Section 9)              |
| 3. Methodology                                | ⚠️ Partially exists | thesis_research_notes.md + Phase1_Analysis.md cover some of this  |
| 4. System Architecture                        | ⚠️ Partially exists | Pipeline diagrams exist but need formal UML/architecture diagrams |
| 5. Data Collection & Preparation              | ❌ Not written      | Corpus provenance, statistics, weak supervision methodology       |
| 6. Phase 1: OCR Correction & CRF Segmentation | ⚠️ Partially exists | Notebooks have content, needs formal write-up                     |
| 7. Phase 2: Transformer-Based Segmentation    | ⚠️ Partially exists | Notebook content needs formal write-up                            |
| 8. Phase 3: Knowledge Graph Safety Guardrail  | ⚠️ Partially exists | Needs formal evaluation                                           |
| 9. Evaluation & Results                       | ❌ Not written      | THE MOST CRITICAL MISSING CHAPTER                                 |
| 10. Discussion                                | ❌ Not written      | Limitations, implications, trade-offs                             |
| 11. Conclusion & Future Work                  | ❌ Not written      | Summary, contributions, future directions                         |
| Appendices                                    | ❌ Not written      | Full KG, suffix lists, sample outputs                             |

---

## 11. RESEARCH PAPER REQUIREMENTS

### For an IEEE/ACL-format research paper:

| Section            | Status | Content Needed                                                |
| ------------------ | ------ | ------------------------------------------------------------- |
| Abstract           | ❌     | 200-word summary of problem, method, results                  |
| Introduction       | ❌     | Problem + 3-4 research questions                              |
| Related Work       | ❌     | 2-3 paragraphs covering SBD, low-resource NLP, medical safety |
| Methodology        | ⚠️     | Formalize the 3-stage pipeline with equations                 |
| Experimental Setup | ❌     | Dataset descriptions, hyperparameters, baselines              |
| Results            | ❌     | Tables with P/R/F1, statistical tests, learning curves        |
| Discussion         | ❌     | Analysis of when/why CRF > RoBERTa, safety implications       |
| Conclusion         | ❌     | Summary + limitations + future work                           |
| References         | ❌     | 25-40 properly formatted citations                            |

---

## 12. DEMO IMPLEMENTATION GAPS

### 12.1 Missing `ayurvedic_segmenter.pkl`

The Streamlit app **cannot run** without the trained CRF model file.

### 12.2 Missing End-to-End Flow

The Streamlit app only demos Phase 1B + 1C (segmentation + safety). It does not include:

- Phase 1A (OCR post-correction via Viterbi)
- Phase 2 (RoBERTa comparison)
- Side-by-side CRF vs. RoBERTa comparison

### 12.3 Missing Visualization

For a compelling demo, the app should include:

- Highlighted sentence boundaries (color-coded)
- Knowledge graph visualization (network diagram of toxic entities)
- Side-by-side CRF vs. RoBERTa output comparison
- Latency comparison chart
- Confidence score visualization (per-word STOP probabilities)

### 12.4 Missing Export Functionality

The demo should be able to:

- Export safety reports as PDF/HTML
- Log all analyses for reproducibility
- Allow batch processing of multiple recipes

---

## 13. ACTIONABLE RECOMMENDATIONS (Priority-Ranked)

### 🔴 PRIORITY 1: CRITICAL (Must fix before any submission)

| #   | Action                                                                                                                   | Effort   |
| --- | ------------------------------------------------------------------------------------------------------------------------ | -------- |
| 1   | **Fix feature mismatch bug** — Unify suffix lists across training, inference, and weak supervision to ONE canonical list | 1 hour   |
| 2   | **Create gold-standard test set** — Manually annotate 300-500 sentences with a domain expert                             | 2-3 days |
| 3   | **Run formal evaluation** — Generate P/R/F1, confusion matrices for both CRF and RoBERTa                                 | 1 day    |
| 4   | **Generate and commit ayurvedic_segmenter.pkl** — Re-run Phase 1 notebook and save the model                             | 1 hour   |
| 5   | **Implement ablation studies** — CRF with/without key features, RoBERTa at multiple data sizes                           | 2 days   |
| 6   | **Add statistical significance tests** — McNemar + bootstrap CI for CRF vs. RoBERTa comparison                           | 1 day    |

### 🟡 PRIORITY 2: IMPORTANT (Must fix before thesis defense)

| #   | Action                                                                          | Effort  |
| --- | ------------------------------------------------------------------------------- | ------- |
| 7   | **Document data provenance** — Where did the 70K corpus come from?              | 1 day   |
| 8   | **Add baseline comparisons** — Rule-based, random, majority class               | 1 day   |
| 9   | **Fix Viterbi log-space** — Convert multiplicative scoring to log-additive      | 2 hours |
| 10  | **Error propagation analysis** — Measure how OCR errors compound downstream     | 2 days  |
| 11  | **Cross-validation** — Replace single 80/20 split with 5-fold CV                | 1 day   |
| 12  | **Knowledge graph source citations** — Add references for every toxicity claim  | 3 days  |
| 13  | **Add Unicode normalization** — NFC normalize all text inputs                   | 2 hours |
| 14  | **Write the evaluation chapter** — Formal tables, figures, statistical analysis | 1 week  |

### 🟢 PRIORITY 3: ENHANCEMENT (Would strengthen the thesis)

| #   | Action                                                                                  | Effort |
| --- | --------------------------------------------------------------------------------------- | ------ |
| 15  | **Implement fuzzy entity matching** — Handle spelling variations in safety guardrail    | 2 days |
| 16  | **Add negation detection** — Prevent false alarms for "do NOT use" instructions         | 2 days |
| 17  | **Add confidence calibration** — Reliability diagrams for CRF probabilities             | 1 day  |
| 18  | **Extend demo with OCR stage** — Show full pipeline including Viterbi                   | 2 days |
| 19  | **Add visualization** — KG network diagram, highlighted sentences, confidence heatmap   | 3 days |
| 20  | **Implement CRF vs. RoBERTa side-by-side demo** — Users see both outputs simultaneously | 1 day  |

---

## APPENDIX A: Suffix List Discrepancy Summary

| Source                                           | Suffixes                                                                          | Count |
| ------------------------------------------------ | --------------------------------------------------------------------------------- | ----- |
| Phase 1 Notebook (Training `word2features`)      | යි, ස්, යුතු, යේය, වේ, මැනවි, ගනු, පෙර, පසු                                       | 9     |
| pipeline.py (Inference `word2features`)          | යි, ස්, යුතු, යේය, වේ, මැනවි, ගනු, පෙර, පසු, කරයි, න්න, ගන්න, තබන්න               | 13    |
| weak_supervision_generator.py (Label generation) | යි, ස්, යුතු, යේය, වේ, මැනවි, ගනු, පෙර, පසු, කරයි, න්න, ගන්න, තබන්න, වස්, නෑ, නැත | 16    |

**Resolution:** Create ONE canonical list and use it everywhere. The training features and inference features MUST match. The weak supervision list can be a superset but should be documented as such.

## APPENDIX B: Recommended Canonical Suffix List

Based on Sinhala linguistic analysis, the canonical list should be:

```python
CANONICAL_ENDINGS = [
    # Finite verb endings (declarative)
    'යි',    # -yi: present tense declarative
    'වේ',    # -wē: copula/become
    'යේය',   # -yēya: past tense classical

    # Imperative/prescriptive (clinical instructions)
    'මැනවි',   # manawi: "it is good to" (formal imperative)
    'ගනු',    # ganu: "take" (imperative)
    'තබන්න',  # thabanna: "keep" (colloquial imperative)
    'කරන්න',  # karanna: "do" (colloquial imperative)  [NOTE: currently 'න්න']
    'ගන්න',   # ganna: "take" (colloquial)
    'කරයි',   # karayi: "does" (present)

    # Obligation markers
    'යුතු',   # yuthu: "must/should"

    # Temporal postpositions (clause-final in Ayurvedic sequences)
    'පෙර',    # pera: "before"
    'පසු',    # pasu: "after"

    # Suffix patterns
    'ස්',     # -s: various verb finals
    'න්න',   # -nna: informal imperative endings

    # Additional patterns from weak supervision
    'වස්',    # was: "for the purpose of"
    'නෑ',     # nǣ: "not" (negative)
    'නැත',    # natha: "not" (formal negative)
]
```

---

_End of Audit Report_
