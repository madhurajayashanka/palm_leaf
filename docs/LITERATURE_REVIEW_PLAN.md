# Literature Review and Research Action Plan

This document serves as your central hub for conducting the literature review and tracking the overall research progress for your project: **A Neuro-Symbolic Pipeline for Sentence Boundary Detection and Toxicity Validation in Archaic Sinhala Ayurvedic Manuscripts**.

---

## 📚 Part 1: Literature Review Focus Areas

Based on your project's unique combination of challenges, you should organize your literature review into the following key themes. For each paper you read, try to map it to one of these areas.

### 1. Low-Resource NLP & Sinhala Processing
*   **Goal:** Understand the state-of-the-art for languages with minimal training data, specifically morphologically rich Indo-Aryan languages.
*   **Key Topics:** Sinhala SBD, morphological analysis of agglutinative languages, transfer learning vs. traditional ML in data-scarce environments.
*   **Search Keywords:** `low-resource NLP sentence boundary detection`, `Sinhala NLP`, `morphologically rich SBD`, `South Asian NLP palm leaf`.

### 2. Sentence Boundary Detection (SBD) without Punctuation
*   **Goal:** Investigate how boundaries are detected when orthographic markers (periods, commas) are completely absent.
*   **Key Topics:** Sequence labeling for SBD, CRF vs. Transformers (RoBERTa/XLM-R) for segmentation, SBD in historical manuscripts or speech transcripts.
*   **Search Keywords:** `SBD without punctuation`, `CRF vs Transformer sequence labeling`, `historical document segmentation`, `speech transcript SBD`.

### 3. Knowledge Graphs (KG) & Safety-Critical Medical AI
*   **Goal:** Review how deterministic rules and KGs are integrated with probabilistic neural outputs to guarantee safety in medical/clinical settings.
*   **Key Topics:** Pharmacovigilance NLP, medical KGs, cascading failures in NLP pipelines, human-in-the-loop AI for clinical safety.
*   **Search Keywords:** `medical knowledge graphs NLP`, `neuro-symbolic AI clinical safety`, `toxicity validation NLP`, `human-in-the-loop medical AI`.

### 4. OCR Post-Correction in High-Noise Environments
*   **Goal:** Review statistical and neural methods for correcting OCR output when no clean parallel corpus exists.
*   **Key Topics:** Noisy channel model, Viterbi decoding, bigram/n-gram language models for OCR correction.
*   **Search Keywords:** `unsupervised OCR post-correction`, `noisy channel OCR historical documents`, `Viterbi decoding OCR`.

---

## 📝 Part 2: Literature Review Paper Template

*Duplicate this template for every significant paper you review.*

### [Paper Title]
*   **Authors:**
*   **Year / Venue:**
*   **Link/DOI:** 

#### 1. Core Problem Addressed
*What problem are the authors trying to solve? How does it relate to our SBD, OCR, or Safety problems?*
> 

#### 2. Proposed Solution / Methodology
*What is their approach? (e.g., CRF, XLM-RoBERTa, rule-based)*
> 

#### 3. Key Findings & Results
*What did they achieve? What metrics did they use?*
> 

#### 4. Limitations
*What did they fail to address? (e.g., requires large data, ignores safety, doesn't work for morphologically rich languages)*
> 

#### 5. Relevance to Our Project
*How will we use this? (e.g., "Provides a baseline for SBD", "Justifies our use of CRF over RoBERTa for <15k sentences", "Informs our KG sliding window design")*
> 

---

## 📋 Part 3: Research & Methodology Checklist

Use this checklist to ensure your research process is scientifically rigorous and ready for publication/thesis submission.

### Phase 1: Literature & Groundwork
- [ ] **Initial Lit Search:** Collect 15-20 core papers across the 4 focus areas listed above.
- [ ] **Gap Analysis:** Clearly articulate the "research gap" (e.g., no existing work addresses SBD + toxicity safety simultaneously in archaic Sinhala).
- [ ] **Data Verification:** Ensure your 70k sentence corpus and 2,100 KG entities are properly documented (sources, annotation guidelines, weak supervision heuristics).

### Phase 2: Methodology Checks
- [ ] **Baseline Establishment:** Define clear baselines (e.g., standard CRF without morphological features vs. XLM-RoBERTa base).
- [ ] **Ablation Studies Design:** 
    - [ ] Test CRF *with* vs. *without* `is_common_ending` feature.
    - [ ] Test RoBERTa learning curve (5k, 15k, 30k, 50k, 70k data points).
    - [ ] Test KG guardrail with varying context window sizes ($k=0, 1, 2, 3$).
- [ ] **Metric Definition:** Confirm metrics are appropriate for highly imbalanced SBD (F1-score for STOP token, not just overall accuracy) and safety (False Negative Rate is critical).

### Phase 3: Experiment Execution
- [ ] **Data Splits:** Ensure strict, non-overlapping Train/Validation/Test splits. Ensure Gold Test set is manually verified.
- [ ] **Run Pipeline Components:** Execute OCR correction, CRF/RoBERTa segmenters, and KG validator.
- [ ] **Store Results:** Save all evaluation metrics systematically (e.g., in `evaluation/results/`).

### Phase 4: Analysis & Writing
- [ ] **Error Analysis:** Manually inspect false positives and false negatives from the KG Guardrail. Why did they fail? (e.g., OCR error vs. SBD error).
- [ ] **Cascading Failure Proof:** Document specific examples where an SBD error directly caused a safety validation failure.
- [ ] **Statistical Significance:** Run McNemar’s test or paired t-tests to prove your hybrid CRF is statistically better than baselines in the low-resource setting.
- [ ] **Drafting:** Write up findings directly into the `docs/THESIS.md` chapters.