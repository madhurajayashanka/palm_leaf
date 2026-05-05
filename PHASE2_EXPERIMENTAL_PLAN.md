# Phase 2: XLM-RoBERTa Comparative Analysis — Research Plan & Hypotheses

**Date:** April 2026  
**Objective:** Test whether multilingual transformers can outperform the Phase 1 CRF baseline on archaic Sinhala sentence segmentation  
**Status:** Pre-execution planning  

---

## 1. Motivating Questions

1. **Does generic multilingual pretraining help?**  
   - XLM-RoBERTa was pretrained on 100+ languages including Sinhala
   - But Sinhala pretraining data is from modern news/web (not archaic Ayurvedic medical texts)
   - **Question:** Can the model transfer this general Sinhala knowledge to archaic domain?

2. **Is 1.2M tokens enough to fine-tune a 300M parameter model?**  
   - CRF trains on 1.2M tokens with ~5K parameters
   - RoBERTa has 300M+ parameters; conventionally requires millions of examples
   - **Question:** Will we overfit, or can we use regularization (dropout, early stopping) to generalize?

3. **Can ensemble methods combine CRF + RoBERTa strengths?**  
   - CRF: Fast, interpretable, leverages morphological domain knowledge
   - RoBERTa: Contextually rich, captures long-range dependencies
   - **Question:** Can we build a voting ensemble that exceeds both?

4. **How much data do we actually need for RoBERTa?**  
   - Phase 1 uses 1.2M tokens (70K sentences)
   - Can we scale to 5M+ tokens by combining multiple Ayurvedic corpora?
   - **Question:** Is the performance plateau due to data scarcity or architectural mismatch?

---

## 2. Core Hypotheses

### Hypothesis H1: Generic Pretraining Doesn't Transfer (Phase 1 > Phase 2)
**Prediction:** XLM-RoBERTa will underperform CRF on archaic Sinhala due to domain mismatch.

**Rationale:**
- RoBERTa's Sinhala corpus is modern web text (news, social media)
- Archaic Ayurvedic Sinhala has different morphology, syntax, vocabulary
- Only 1.2M tokens to fine-tune on; likely insufficient for 300M-parameter model

**Expected Outcomes:**
- XLM-RoBERTa Exp1 (frozen BERT, fine-tune LSTMs): ~70% F1(STOP)
- XLM-RoBERTa Exp2 (full fine-tuning): ~72% F1(STOP)
- XLM-RoBERTa Exp3 (LoRA): ~75% F1(STOP)
- All significantly below CRF baseline (82.68% F1(STOP))

**Validation Criteria:**
- RoBERTa F1(STOP) < CRF F1(STOP) - 5% (i.e., <77.68%)
- Confirms that domain engineering beats generic pretraining for low-resource archaic languages

---

### Hypothesis H2: More Data Enables Competitive Performance (Phase 2 ≈ Phase 1)
**Prediction:** If we scale to 5M+ tokens, RoBERTa can match or exceed CRF.

**Rationale:**
- Transformer models follow scaling laws: more data → better performance
- 5M tokens provides better gradient signal for 300M-parameter model
- RoBERTa's superior context modeling (multi-head attention) may capture complex patterns

**Expected Outcomes:**
- XLM-RoBERTa Exp4 (5M tokens): ~85–88% F1(STOP)
- Competitive with CRF (82.68%), narrowing gap to <2%

**Validation Criteria:**
- RoBERTa F1(STOP) with 5M tokens > 85%
- Learning curve shows continued improvement (no plateau at 1.2M tokens)

---

### Hypothesis H3: Ensemble Approach Achieves Best Results (Exp5 Hybrid)
**Prediction:** Voting ensemble of CRF + RoBERTa exceeds both individual models.

**Rationale:**
- CRF captures morphological patterns; RoBERTa captures context
- Complementary strengths: ensemble voting can correct both models' mistakes
- Prior work on sequence labeling shows ensemble gains of 1–3% F1

**Expected Outcomes:**
- Ensemble (CRF + RoBERTa): ~88–90% F1(STOP)
- Exceeds best single model by 2–3% F1

**Validation Criteria:**
- Ensemble F1(STOP) > max(CRF, RoBERTa) + 1%
- Error analysis shows ensemble fixes different mistake types than either model

---

## 3. Experimental Design

### 3.1 Experiment Hierarchy

```
Phase 2 Experiments
│
├─ Exp1: XLM-RoBERTa (Frozen BERT, Fine-tune LSTMs)
│  └─ Data: 1.2M tokens
│  └─ Params trainable: ~2M (LSTM only)
│  └─ Expected F1(STOP): ~70%
│
├─ Exp2: XLM-RoBERTa (Full Fine-tuning)
│  └─ Data: 1.2M tokens
│  └─ Params trainable: 300M+
│  └─ Regularization: Dropout=0.1, Early stop on val F1
│  └─ Expected F1(STOP): ~72%
│
├─ Exp3: XLM-RoBERTa (LoRA Fine-tuning)
│  └─ Data: 1.2M tokens
│  └─ Params trainable: ~100K (LoRA only, r=8)
│  └─ Rationale: Parameter-efficient, reduces overfitting
│  └─ Expected F1(STOP): ~75%
│
├─ Exp4: XLM-RoBERTa (5M+ Tokens)
│  └─ Data sources: Combined Ayurvedic corpora
│  └─ Params trainable: 300M+ (full fine-tuning)
│  └─ Expected F1(STOP): ~85–88%
│
└─ Exp5: Ensemble (CRF + RoBERTa)
   └─ Base models: CRF (Phase 1) + RoBERTa (best from Exp1–4)
   └─ Voting strategy: Token-level majority vote
   └─ Expected F1(STOP): ~88–90%
```

### 3.2 Experiment Details

#### Experiment 1: Frozen BERT + Fine-tuned LSTMs

**Motivation:** Test whether RoBERTa's frozen embeddings alone are sufficient without updating the full model.

**Setup:**
```python
# Pseudocode
model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
model.embeddings.requires_grad = False  # Freeze
model.encoder.requires_grad = False     # Freeze

# Add LSTM decoder on top
lstm = nn.LSTM(768, 256, num_layers=2, bidirectional=True)
classifier = nn.Linear(512, 2)  # O, STOP

# Only train LSTM + classifier
```

**Expected Performance:** ~70% F1(STOP)

**Rationale:** Tests whether pretrained embeddings alone provide sufficient linguistic information without updating transformers layers.

---

#### Experiment 2: Full Fine-tuning (1.2M tokens)

**Motivation:** Fine-tune all 300M parameters on 1.2M tokens; test capacity and overfitting.

**Setup:**
```python
model = XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-base')
model.requires_grad = True  # Unfreeze all

# Training
optimizer = AdamW(model.parameters(), lr=2e-5)
trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    epochs=10,
    early_stopping_patience=2,  # Stop if val F1 doesn't improve for 2 epochs
    loss_function=CrossEntropyLoss()
)
```

**Expected Performance:** ~72% F1(STOP)

**Rationale:** Likely to overfit due to small data relative to parameter count. Early stopping should mitigate.

---

#### Experiment 3: LoRA Fine-tuning (1.2M tokens, Parameter-Efficient)

**Motivation:** Use LoRA (Low-Rank Adaptation) to reduce trainable parameters and prevent overfitting.

**Setup:**
```python
# Add LoRA to attention layers
from peft import get_peft_model, LoraConfig

config = LoraConfig(
    r=8,  # Rank of LoRA matrices
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(
    XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-base'),
    config
)
# Only ~100K parameters trainable now (vs 300M)
```

**Expected Performance:** ~75% F1(STOP)

**Rationale:** LoRA reduces overfitting on small datasets while maintaining model capacity.

---

#### Experiment 4: Full Fine-tuning (5M+ tokens)

**Motivation:** Test scaling hypothesis — does more data enable better RoBERTa performance?

**Data Strategy:**
- Combine multiple Ayurvedic sources if available
- Or synthesize additional training data via back-translation (English → Sinhala)
- Target: 5M tokens (~300K sentences)

**Setup:**
```python
# Same as Exp2, but with 5M tokens
trainer = Trainer(
    model=model,
    train_dataset=train_data_large,  # 5M tokens
    eval_dataset=val_data,
    epochs=5,  # Fewer epochs with more data
    early_stopping_patience=2,
)
```

**Expected Performance:** ~85–88% F1(STOP)

**Rationale:** Scaling laws suggest transformer performance improves ~0.01 F1 per 10x data increase.

---

#### Experiment 5: Ensemble (CRF + Best RoBERTa)

**Motivation:** Combine CRF and RoBERTa predictions via voting.

**Setup:**
```python
def ensemble_predict(text, crf_model, roberta_model, voting='soft'):
    """
    For each word, predict STOP probability using both models.
    Soft voting: average probabilities
    Hard voting: majority vote
    """
    crf_probs = crf_model.predict_marginals([text])
    roberta_probs = roberta_model.predict_probabilities([text])
    
    if voting == 'soft':
        ensemble_probs = (crf_probs + roberta_probs) / 2
    else:
        crf_vote = (crf_probs['STOP'] > 0.15).astype(int)
        roberta_vote = (roberta_probs['STOP'] > 0.5).astype(int)
        ensemble_probs = {'STOP': (crf_vote + roberta_vote) / 2}
    
    return ensemble_probs
```

**Expected Performance:** ~88–90% F1(STOP)

**Rationale:** CRF + RoBERTa have complementary strengths; voting combines them.

---

### 3.3 Evaluation Metrics

For each experiment, measure:

| Metric | Purpose | Threshold |
|--------|---------|-----------|
| F1(STOP) | Segment detection accuracy | >75% for Phase 2 success |
| F1(O) | Baseline/continue accuracy | Usually >95% |
| Precision(STOP) | False positive rate | >80% (minimize wrong boundaries) |
| Recall(STOP) | False negative rate | >70% (catch most real boundaries) |
| Accuracy | Overall token classification | >94% |
| Latency (ms/sentence) | Inference speed | <10ms for deployment |
| GPU Memory (GB) | Hardware requirements | <8GB for laptop deployment |

---

### 3.4 Evaluation Strategy

**Train/Val/Test Split:**
- **Training:** 70% of labeled data (~49K sentences, 0.84M tokens)
- **Validation:** 15% (~10.5K sentences, 0.18M tokens) — used for early stopping
- **Test:** 15% (~10.5K sentences, 0.18M tokens) — held out; final evaluation

**Cross-Validation (Optional):**
- 5-fold cross-validation on full dataset to estimate confidence intervals
- Report mean ± std F1(STOP)

**Statistical Significance:**
- McNemar's test to compare RoBERTa vs CRF on binary decisions (STOP vs O)
- If p > 0.05, no statistically significant difference

---

## 4. Expected Results Matrix

### Performance Summary

```
Experiment                   F1(STOP)    Latency    GPU?    Parameters
─────────────────────────────────────────────────────────────────────
Phase 1: CRF Baseline        82.68%      0.04ms     ❌      5K
Exp1: Frozen BERT            ~70%        2ms        ❌      2M
Exp2: Full FT (1.2M)         ~72%        3ms        ✅      300M
Exp3: LoRA FT (1.2M)         ~75%        3ms        ✅      100K
Exp4: Full FT (5M)           ~85%        3ms        ✅      300M
Exp5: Ensemble (CRF+best)    ~88%        5ms        ✅      ~100M

Target for Phase 2 Success: F1(STOP) ≥ 75%
```

### Failure Modes & Contingencies

| If This Happens | Action |
|-----------------|--------|
| RoBERTa F1(STOP) < 70% on all exps | Hypothesis H1 confirmed; recommend sticking with CRF; publish finding that domain eng > generic PT |
| RoBERTa F1(STOP) > 85% on Exp4 only | Hypothesis H2 confirmed; need 5M+ tokens; recommend data collection effort |
| Ensemble doesn't exceed best model | Try different voting strategies (weighted vote, learned combiner); may not provide gains |
| Out-of-memory errors during training | Reduce batch size, use gradient accumulation, or switch to inference-only approach |
| Validation F1 plateaus early | May indicate data insufficiency or model architectural mismatch; try different architectures |

---

## 5. Phase 2 Deliverables

### Code & Artifacts
- [ ] Fine-tuning scripts for Exp1–Exp5 (Python + Hugging Face)
- [ ] Evaluation scripts (metrics, confusion matrices, significance tests)
- [ ] Ensemble inference pipeline
- [ ] Pretrained model checkpoints (best RoBERTa + ensemble)

### Experimental Reports
- [ ] Experiment 1–5 results (F1, latency, GPU memory)
- [ ] Error analysis (examples where RoBERTa succeeds/fails vs CRF)
- [ ] Learning curves (F1 vs training data size)
- [ ] Ablation study (which attention heads matter? which layers?)

### Publication-Ready Outcomes
- [ ] Comparative table: CRF vs RoBERTa performance
- [ ] Statistical significance test results (McNemar's)
- [ ] Recommendations for practitioners (when to use each model)
- [ ] Replicable code repository (GitHub with Jupyter notebooks)

### Updated Documentation
- [ ] Phase 2 Research Paper (extending Phase 1 thesis)
- [ ] Lessons learned: Transformers vs classical ML for low-resource NLP
- [ ] Deployment guide for Phase 1 CRF (now that Phase 2 evaluated alternatives)

---

## 6. Timeline & Resource Planning

| Phase | Timeline | Resources | Output |
|-------|----------|-----------|--------|
| Setup | Week 1 | GPU (RTX 3080+), Hugging Face | Scripts, environment |
| Exp1–3 | Week 2–3 | GPU, 1.2M token data | 3 model checkpoints |
| Exp4 | Week 4–5 | GPU, 5M+ token data | 1 model checkpoint, scaling curve |
| Exp5 | Week 5 | CPU (ensemble inference) | Ensemble model, voting strategy |
| Analysis | Week 6 | CPU, Jupyter | Plots, tables, significance tests |
| Writing | Week 7–8 | CPU | Phase 2 report, recommendations |

**Total Duration:** 8 weeks (assuming part-time work)

---

## 7. Success Criteria & Decision Framework

### If H1 is Confirmed (RoBERTa < CRF)
**Conclusion:** "Domain engineering beats generic pretraining for archaic languages."

**Recommendation:**
- Publish finding in NLP venue (e.g., ACL, EMNLP)
- Recommend **Phase 1 CRF for production deployment**
- Justify thesis: Low-resource languages benefit from hand-crafted features, not scaling up generic models
- Implications for NLP community: Reconsider transformer-first approach for endangered languages

**Citation:** "For archaic Sinhala Ayurvedic texts, a hybrid CRF with morphological features (82.68% F1) outperforms XLM-RoBERTa fine-tuning (75% F1), suggesting that domain engineering is more effective than generic multilingual pretraining for low-resource historical NLP."

---

### If H2 is Confirmed (RoBERTa ≥ CRF with 5M tokens)
**Conclusion:** "Scaling is the bottleneck, not architecture."

**Recommendation:**
- Invest in data collection (combine multiple Ayurvedic corpora, digitization projects)
- Once 5M+ tokens available, transition to RoBERTa for production
- Hybrid approach: Use CRF for low-latency inference, RoBERTa for batch processing

**Citation:** "XLM-RoBERTa with 5M+ tokens of domain data achieves competitive performance (85% F1), validating the hypothesis that transformer scaling laws apply to morphologically complex low-resource languages."

---

### If H3 is Confirmed (Ensemble > 88% F1)
**Conclusion:** "Complementary models provide additive value."

**Recommendation:**
- Deploy ensemble as best-effort solution
- Slightly higher latency (5ms vs 3ms) acceptable for medical safety-critical system
- Provides interpretability (if CRF says STOP but RoBERTa says O, needs human review)

**Citation:** "An ensemble of CRF and RoBERTa achieves 88% F1(STOP), demonstrating that neuro-symbolic hybrid approaches (combining domain engineering with deep learning) are superior to either component alone for safety-critical medical NLP."

---

## 8. Risk Assessment & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| RoBERTa overfits on 1.2M tokens | High | Exp1–3 fail to train properly | Use LoRA (Exp3), early stopping, dropout |
| GPU out of memory | Medium | Can't run Exp2, Exp4 | Reduce batch size, use gradient accumulation |
| No improvement with more data (Exp4) | Low | Invalidates H2 | Suggests architectural ceiling; switch to other models |
| Ensemble doesn't help (Exp5) | Medium | False hope for gains | Document complementarity; may be inherent limits |
| Sinhala pretraining data quality | Low | RoBERTa embeddings noisy | Collect gold-standard Sinhala validation set |

---

## 9. Open Questions

1. **Does archaic Sinhala have sufficient representation in XLM-RoBERTa's pretraining?**  
   - If yes: Exp1–3 should show reasonable performance
   - If no: RoBERTa embeddings may be noisy; consider LoRA (Exp3) or retrain from scratch

2. **Can we synthesize more training data via back-translation?**  
   - Translate English Ayurvedic texts → Sinhala to boost corpus
   - Trade-off: synthetic data quality vs quantity

3. **What patterns does RoBERTa learn that CRF misses?**  
   - Error analysis: compare mistakes on same sentences
   - Gain insights into what transformers add (multi-hop reasoning? long-range deps?)

4. **Can we fine-tune RoBERTa on Sinhala-specific tasks first (e.g., POS tagging) before segment segmentation?**  
   - Multi-task learning approach
   - May improve Ayurvedic domain adaptation

---

## 10. References & Related Work

- Devlin et al. (2018): BERT — Foundation for RoBERTa
- Hu et al. (2021): LoRA — Parameter-efficient fine-tuning
- Conneau et al. (2019): XLM-RoBERTa — Multilingual pretraining
- Lafferty et al. (2001): CRF — Conditional Random Fields
- Huang et al. (2015): Bidirectional LSTM-CRF for sequence labeling

---

**Prepared by:** PhD Research Team  
**Date:** April 2026  
**Status:** Ready for Phase 2 execution
