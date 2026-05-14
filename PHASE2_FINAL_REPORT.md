# Phase 2 — Module 3 — Final Report

**Date:** 2026-05-14
**Module:** 3 — SBD, OCR Post-Correction, KG Safety, Pipeline Orchestration, UI
**Status:** Code complete · awaiting Colab training run

---

## 1. What changed since Phase 1

Phase 1 was audited and found to have four severe methodological flaws
([docs/FULL_AUDIT_REPORT.md](docs/FULL_AUDIT_REPORT.md)):

1. **Random gold standard** — `scripts/generate_gold.py` was a 20-sentence
   hand-typed set; the older version had used random labels. Either way,
   the test set was too small and structurally trivial (single end-of-line
   STOP per sequence).
2. **Circular weak supervision** — the CRF was trained on labels produced
   by exactly the same suffix rule it received as a feature
   (`is_common_ending`). The audit's ablation confirmed `word[-2:]` already
   captures the rule, so the "hybrid" claim was empirically dead.
3. **Unfair RoBERTa comparison** — both models were measured against the
   noisy random-rule gold and on labels that perfectly favoured the CRF.
4. **8-string safety eval** — clinically meaningless.

Phase 2 of Module 3 fixes every one of these and adds three genuine
research contributions on top.

---

## 2. Module 3 contributions (Phase 2)

| # | Contribution | Files | Fixes audit flaw | Novelty |
|---|---|---|---|---|
| **C1** | **Multi-rule probabilistic weak supervision (Snorkel-lite)** — six independent labelling functions (line-end anchor, canonical suffix, extended suffix, clinical-verb membership, negation, **data-driven `P(end | word)` from corpus statistics**) combined through a generative log-odds label model that emits *soft* labels in [0, 1]. | [src/labeling_functions.py](src/labeling_functions.py), [src/label_model.py](src/label_model.py), [scripts/build_soft_training_set.py](scripts/build_soft_training_set.py) | Flaw 2 (circular WS) | First Snorkel-style WS pipeline for archaic Sinhala SBD |
| **C2** | **Domain-Adaptive Pretraining (DAPT) + Morphology-Injected Classification Head** — continued MLM on `cleaned_corpus.txt` before SBD fine-tuning, plus a custom head that concatenates the 16-dim hand-crafted morphology vector with the XLM-RoBERTa contextual embedding. | [src/morphology_features.py](src/morphology_features.py), [notebooks/phase2/Phase2_DAPT_and_Finetune.ipynb](notebooks/phase2/Phase2_DAPT_and_Finetune.ipynb) | Flaw 3 (unfair comparison) | First DAPT for archaic Sinhala + novel morphology-injected head architecture |
| **C3** | **Confidence-cascading pipeline with uncertainty propagation** — every segmenter emits a sequence-level reliability score; the safety guardrail's verdict has three classes (APPROVE / REJECT / **HITL**); HITL fires when reliability < τ and a toxic match is present. | [src/confidence_pipeline.py](src/confidence_pipeline.py) | Cascading-failure issue identified in thesis §5.1 | Novel HITL escalation for medical NLP |
| **C4** | **Multi-bucket gold standard (315 sequences) + 70-scenario KG safety benchmark** — gold sampled from real corpus, stratified into EASY / MED / HARD / EXPERT difficulty buckets with multi-boundary sequences; safety benchmark stress-tests 10 distinct scenario kinds including alias resolution, plant-part variation, distant shodhana, and negated shodhana. | [scripts/build_gold_v2.py](scripts/build_gold_v2.py), [scripts/build_safety_benchmark.py](scripts/build_safety_benchmark.py) | Flaws 1 + 4 | Publishable benchmark resource |
| **C5** | **Fair evaluation framework** — every SBD model compared against the rule-only baseline (the *primary* baseline required by the audit's Step 2) on identical gold with McNemar significance, bootstrap CIs, per-bucket breakdown, and an artificial-perturbation cascade-failure sweep. | [evaluation/evaluate_phase2.py](evaluation/evaluate_phase2.py) | All 4 | Honest comparative analysis |
| **+** | **Updated Streamlit UI** — pipeline selector, per-token P(STOP) heatmap, HITL τ slider, APPROVE / REJECT / HITL verdict surface. | [app/app_phase2.py](app/app_phase2.py) | — | HITL UX |

---

## 3. Critical issue I found — and what it means

While verifying the corpus, I discovered the "70,000 sentence Sinhala
Ayurvedic corpus" (`data/cleaned_corpus.txt`) contains **only 205 unique
word types across 782,140 tokens**. The bigram model has 197 vocab
entries and 3,824 pairs. The Phase 1 summary's claim of "50,000+ unique
word types / 350,000+ bigram pairs" is incorrect.

**Implication:** the corpus is essentially a templated/synthetic
recipe-text generator. Every model trained on it will look very strong on
in-corpus evaluation because the combinatorial complexity is small.

**How I'm handling this for Module 3:**

1. Module 3's *methodological* contributions (C1–C5) remain valid: they
   make the pipeline scientifically honest, and they generalise to richer
   corpora.
2. The Phase 2 report (this document, §6) frames numerical results
   correctly: *methodology feasibility on a constrained corpus*, with a
   clear "future work: scale up corpus" call-out.
3. The **EXPERT** bucket in [gold_test_v2.tsv](data/gold_test_v2.tsv)
   contains hand-typed authentic Ayurvedic sentences *not* drawn from the
   templated corpus, providing a small out-of-distribution test signal.
4. **Recommendation:** for the final paper, collect a second corpus
   (digitised real palm-leaf scans, or scrape modern Ayurvedic
   reference texts) so the numerical claims rest on a more realistic
   base. This is a Module 1 / data-acquisition task and a fair scoping
   choice for future work.

This finding does *not* affect Module 3's individual contributions, but
you should know about it before you defend the numbers in the viva.

---

## 4. What you actually do — step-by-step

All the code is written. You only need to run two notebooks on Colab
and one evaluation script locally.

### Step 4.1  Verify locally (already passing on my end)
```bash
# Already produced these:
ls data/train_soft.tsv data/train_hard.tsv data/train_soft_meta.json \
   data/gold_test_v2.tsv data/safety_benchmark.jsonl data/endword_statistics.json
```

If any of those is missing, rebuild it:
```bash
python scripts/build_soft_training_set.py --n-sequences 12000
python scripts/build_gold_v2.py
python scripts/build_safety_benchmark.py
```

### Step 4.2  Push to GitHub
The Phase 2 Colab notebook clones from `https://github.com/madhurajayashanka/palm_leaf.git`.
**Push these new files first** so Colab can pick them up:

```bash
git add src/labeling_functions.py src/label_model.py src/morphology_features.py src/confidence_pipeline.py \
        scripts/build_soft_training_set.py scripts/build_gold_v2.py scripts/build_safety_benchmark.py \
        evaluation/evaluate_phase2.py \
        notebooks/phase2/Phase2_DAPT_and_Finetune.ipynb \
        app/app_phase2.py \
        data/train_soft.tsv data/train_hard.tsv data/train_soft_meta.json \
        data/gold_test_v2.tsv data/gold_test_v2.meta.json \
        data/safety_benchmark.jsonl data/endword_statistics.json \
        PHASE2_FINAL_REPORT.md
git commit -m "Phase 2 Module 3: multi-LF WS, DAPT, morph head, HITL cascade, fair eval"
git push origin main
```

### Step 4.3  Train on Colab (~2 h on free T4)
Open `notebooks/phase2/Phase2_DAPT_and_Finetune.ipynb` in Colab, switch
runtime to GPU, and run all cells. Three checkpoints land in
`data/models/`:
- `dapt_xlmr/` — MLM-adapted encoder
- `sbd_xlmr_baseline/` — vanilla XLM-R fine-tuned for SBD
- `sbd_xlmr_dapt/` — DAPT-then-fine-tune
- `sbd_xlmr_dapt_morph/` — DAPT + morphology-injected head (**the novel model**)

Plus `data/models/phase2_training_summary.json` with the val metrics.

**Download** all of `data/models/` back to your local machine (right-click
the folder in Colab → Download as zip). Unzip into `data/models/`.

### Step 4.4  Run fair evaluation locally
```bash
python evaluation/evaluate_phase2.py \
       --include-crf \
       --include-baseline-roberta \
       --include-dapt-roberta \
       --include-dapt-morph
```

This writes `evaluation/results/phase2_eval.json` with:
- per-model F1(STOP), Precision, Recall, Accuracy
- per-bucket breakdown (EASY / MED / HARD / EXPERT)
- McNemar's test of every model vs. the rule-only baseline
- bootstrap 95% CI on F1(STOP)
- safety-benchmark accuracy per scenario kind per window
- cascade-failure sweep at perturbation rates 0–50%

### Step 4.5  Launch the new Streamlit demo
```bash
streamlit run app/app_phase2.py
```

The sidebar auto-detects whichever checkpoints exist in `data/models/`.

---

## 5. Headline numbers I can confirm without GPU training

Already run locally on rule_only + legacy CRF against the new gold v2:

```
model            F1(STOP)   Precision   Recall   Acc    F1 CI95
─────────────────────────────────────────────────────────────────
rule_only          0.544      0.659     0.464   0.931   [0.517, 0.571]
crf (legacy)       0.994      1.000     0.989   0.999   [0.991, 0.997]
```

### 🔑 Critical per-bucket finding (this is the headline of your paper)

When sliced by difficulty bucket, the picture *completely inverts* on
the EXPERT bucket — the only bucket containing authentic Ayurvedic
sentences not drawn from the templated training corpus:

| Bucket | rule_only F1 | CRF F1 | Interpretation |
|---|---|---|---|
| EASY (in-corpus, 2 sents)  | 0.535 | **1.000** | CRF perfectly memorises in-corpus patterns |
| MED  (in-corpus, 3-4 sents) | 0.546 | **1.000** | Same |
| HARD (in-corpus, 5-7 sents) | 0.541 | **1.000** | Same |
| **EXPERT (out-of-corpus authentic)** | **0.769** | **0.235** | **Rule baseline beats CRF by 53 absolute F1 points** |

**This empirically confirms the audit's concern:** the CRF's 99.4%
in-corpus F1 is template-memorisation, not generalisation. On real
Ayurvedic text outside the training distribution the morphology-rule
baseline outperforms the CRF by a huge margin.

McNemar test, CRF vs rule_only across the whole gold:
χ² = 876.3, **p = 1.4 × 10⁻¹⁹²**. Both directions of disagreement:
- CRF correct / rule wrong: 902 tokens (mostly in-corpus)
- rule correct / CRF wrong: 8 tokens (mostly EXPERT)

### What this means for Module 3's research story

This is now the *strongest* possible motivation for the DAPT + morphology-injected
RoBERTa model:

> The legacy CRF, while reporting 99.4% in-corpus F1, generalises poorly
> (23.5% F1 on out-of-corpus Ayurvedic sentences) because its features
> are tied to a small templated vocabulary. The rule-only baseline
> generalises better (76.9% F1) because morphological suffixes are
> register-invariant. We hypothesise that DAPT-adapted XLM-RoBERTa with
> a morphology-injected head will dominate the EXPERT bucket without
> sacrificing in-corpus accuracy.

That hypothesis is *testable*. The answer (whether DAPT-Morph wins) is
genuinely uncertain. You'll know after the Colab run.

---

## 6. Result template — fill after Colab run

(I'll fill these in once you upload the training results JSON.)

### 6.1  SBD comparative table

| Model | F1(STOP) | P | R | Acc | F1 EASY | F1 MED | F1 HARD | F1 EXPERT | McNemar vs rule-only |
|---|---|---|---|---|---|---|---|---|---|
| rule_only            | 0.544 | 0.659 | 0.464 | 0.931 | _TBC_ | _TBC_ | _TBC_ | _TBC_ | — |
| CRF (legacy)         | 0.994 | 1.000 | 0.989 | 0.999 | _TBC_ | _TBC_ | _TBC_ | _TBC_ | p = _TBC_ |
| XLM-R baseline       | _TBC_ | _TBC_ | _TBC_ | _TBC_ | _TBC_ | _TBC_ | _TBC_ | _TBC_ | p = _TBC_ |
| XLM-R + DAPT         | _TBC_ | _TBC_ | _TBC_ | _TBC_ | _TBC_ | _TBC_ | _TBC_ | _TBC_ | p = _TBC_ |
| **XLM-R + DAPT + Morph (novel)** | _TBC_ | _TBC_ | _TBC_ | _TBC_ | _TBC_ | _TBC_ | _TBC_ | _TBC_ | p = _TBC_ |

### 6.2  Safety benchmark — per-scenario accuracy at k=1

| Scenario kind | Expected | rule_only | CRF | XLM-R DAPT-Morph |
|---|---|---|---|---|
| SAFE_HERB           | APPROVE | _TBC_ | _TBC_ | _TBC_ |
| TOXIC_PURIFIED      | APPROVE | _TBC_ | _TBC_ | _TBC_ |
| TOXIC_ADJACENT      | APPROVE | _TBC_ | _TBC_ | _TBC_ |
| TOXIC_DISTANT       | REJECT  | _TBC_ | _TBC_ | _TBC_ |
| TOXIC_UNPURIFIED    | REJECT  | _TBC_ | _TBC_ | _TBC_ |
| TOXIC_NEGATED       | REJECT  | _TBC_ | _TBC_ | _TBC_ |
| MULTI_PARTIAL       | REJECT  | _TBC_ | _TBC_ | _TBC_ |
| MULTI_ALL_PURIFIED  | APPROVE | _TBC_ | _TBC_ | _TBC_ |
| ALIAS_PURIFIED      | APPROVE | _TBC_ | _TBC_ | _TBC_ |
| PLANTPART_PURIFIED  | APPROVE | _TBC_ | _TBC_ | _TBC_ |

### 6.3  Cascading-failure decay

```
perturbation rate   0%    5%   10%   20%   30%   50%
─────────────────────────────────────────────────────
rule_only         _TBC_  _TBC_ _TBC_ _TBC_ _TBC_ _TBC_
CRF               _TBC_  _TBC_ _TBC_ _TBC_ _TBC_ _TBC_
XLM-R + DAPT-Morph _TBC_ _TBC_ _TBC_ _TBC_ _TBC_ _TBC_
```

### 6.4  HITL coverage at varying τ

The HITL τ slider trades safety risk for human-review burden.  Fill once
the model is trained:

| τ_HITL | % HITL | False-negative rate | Practitioner burden |
|---|---|---|---|
| 0.30 | _TBC_ | _TBC_ | low |
| 0.55 | _TBC_ | _TBC_ | mid |
| 0.70 | _TBC_ | _TBC_ | high |

---

## 7. Known issues / things to flag to your supervisor

1. **Corpus vocab is 205** — see §3. Genuinely surprising; flag this in viva.
2. **`evaluate.py` (the old script) still claims F1=1.0 for the CRF** because
   it uses 80/20 split of the templated train data with one-sentence
   sequences. Don't quote that script's numbers any more. Use
   `evaluation/evaluate_phase2.py` and `evaluation/results/phase2_eval.json`.
3. **The XLM-R-Morph head uses `pytorch_model.bin`** (full state dict)
   because we can't save a custom head via `model.save_pretrained()`. The
   eval framework rebuilds the same class definition to load it; if you
   change the architecture, update both [Phase2_DAPT_and_Finetune.ipynb](notebooks/phase2/Phase2_DAPT_and_Finetune.ipynb) cell 7 and [evaluation/evaluate_phase2.py](evaluation/evaluate_phase2.py) `make_hf_morph_predictor`.
4. **`evaluation/evaluate.py` (legacy) is now stale** but I left it untouched
   so your team-mates' modules don't break. New work uses
   `evaluate_phase2.py`.

---

## 8. File map of the deliverable

```
src/
  labeling_functions.py        ← C1 — 6 LFs + corpus stats
  label_model.py               ← C1 — Snorkel-lite generative label model
  morphology_features.py       ← C2 — 16-dim hand-crafted morphology vector
  confidence_pipeline.py       ← C3 — uncertainty-propagating cascade

scripts/
  build_soft_training_set.py   ← C1 — emits train_soft.tsv + train_hard.tsv
  build_gold_v2.py             ← C4 — multi-bucket gold (315 sequences)
  build_safety_benchmark.py    ← C4 — 70 KG-safety scenarios

evaluation/
  evaluate_phase2.py           ← C5 — fair-comparison framework + cascade sweep
  results/phase2_eval.json     ← emitted by C5

notebooks/phase2/
  Phase2_DAPT_and_Finetune.ipynb  ← C2 — Colab-ready DAPT + 3 fine-tune variants

app/
  app_phase2.py                ← updated Streamlit (HITL surface)

data/
  train_soft.tsv               ← C1 output, used to train RoBERTa
  train_hard.tsv               ← C1 output, used to train CRF baseline
  train_soft_meta.json         ← C1 metadata (LF accuracies/weights)
  endword_statistics.json      ← C1 cache (P(end|word) from corpus)
  gold_test_v2.tsv             ← C4 — multi-bucket gold
  gold_test_v2.meta.json       ← C4 — per-sequence bucket + ambiguity
  safety_benchmark.jsonl       ← C4 — 70 safety scenarios

PHASE2_FINAL_REPORT.md         ← this file
```
