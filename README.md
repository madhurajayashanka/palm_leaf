# Ayurvedic NLP Pipeline

A neuro-symbolic NLP pipeline for Sinhala Ayurvedic text processing — sentence segmentation, OCR post-correction, and toxicity safety guardrails.

## Project Structure

```
├── src/                        # Core source code
│   ├── config.py               # Central configuration (suffixes, hyperparameters, paths)
│   ├── pipeline.py             # CRF segmenter + knowledge graph safety guardrail
│   └── viterbi_decoder.py      # Viterbi OCR post-correction with bigram LM
│
├── app/                        # Streamlit web applications
│   ├── app.py                  # Simple segmentation + safety UI
│   └── app_demo.py             # Full demo with OCR correction + architecture docs
│
├── evaluation/                 # Evaluation scripts & results
│   ├── evaluate.py             # Comprehensive evaluation (CV, ablation, baselines, latency)
│   ├── evaluate_cascading.py   # Cascading failure error propagation tests
│   ├── evaluate_ocr.py         # OCR post-correction evaluation
│   └── results/                # JSON evaluation outputs
│
├── scripts/                    # Data generation utilities
│   ├── build_bigram_model.py   # Build bigram LM from corpus
│   └── generate_gold.py        # Generate gold-standard test set
│
├── data/                       # Data files
│   ├── ayurvedic_ingredients_full.csv
│   ├── ayurvedic_ingredients_sample.csv
│   ├── bigram_probabilities.json
│   ├── cleaned_corpus.txt
│   ├── train_labeled.tsv
│   ├── train.txt
│   ├── gold_test.tsv
│   └── models/
│       └── ayurvedic_segmenter.pkl
│
├── docs/                       # Research documentation
│   ├── THESIS.md
│   ├── thesis.tex
│   ├── thesis_research_notes.md
│   ├── RESEARCH_PAPER.md
│   ├── research_paper.tex
│   ├── FULL_AUDIT_REPORT.md
│   └── SECOND_AUDIT_REPORT.md
│
├── notebooks/                  # Jupyter experiment notebooks
│   ├── phase1/                 # Phase 1: CRF + Bigram pipeline
│   └── phase2/                 # Phase 2: RoBERTa pipeline
│
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# Build the bigram language model
python scripts/build_bigram_model.py

# Generate gold test set
python scripts/generate_gold.py

# Run evaluation
python evaluation/evaluate.py

# Launch the web app
streamlit run app/app.py
# Or the full demo
streamlit run app/app_demo.py
```
