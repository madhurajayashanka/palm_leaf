"""Phase-2 Streamlit demo — confidence-cascading pipeline with HITL triage.

Run:
    streamlit run app/app_phase2.py

Differences from `app/app.py`:
  * Pipeline selector: legacy CRF *or* DAPT-RoBERTa *or* DAPT-RoBERTa-Morph
    (auto-detects which checkpoints are available under data/models/).
  * Shows per-token P(STOP) heatmap so the practitioner can see which
    boundaries the model is unsure about.
  * Surfaces the HITL verdict from `cascade_safety` when the segmenter is
    not confident enough — this is the *novel* third decision class
    (APPROVE / REJECT / HITL).
  * Lets the practitioner adjust the HITL τ at runtime.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
for d in (ROOT, SRC):
    if d not in sys.path:
        sys.path.insert(0, d)

import streamlit as st

from confidence_pipeline import (
    SegmentationResult,
    cascade_safety,
    load_knowledge_graph,
    segment_with_callable,
    segment_with_crf,
    sequence_reliability,
)
from config import MODELS_DIR

# --------------------------------------------------------------------------
# Lazy / cached model loaders
# --------------------------------------------------------------------------
@st.cache_resource
def get_kg():
    return load_knowledge_graph()


@st.cache_resource
def get_crf():
    import joblib
    path = os.path.join(MODELS_DIR, "ayurvedic_segmenter.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource
def get_hf_token_predictor(model_dir: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    if not os.path.isdir(model_dir):
        return None
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def predict(words):
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
        for ti, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            out[wid] = float(probs[ti])
            seen.add(wid)
        return out
    return predict


def list_available_segmenters() -> dict:
    """Return display-name → factory pairs for whichever models are on disk."""
    segs = {}
    if os.path.exists(os.path.join(MODELS_DIR, "ayurvedic_segmenter.pkl")):
        segs["CRF (legacy)"] = ("crf", None)
    for d, name in [
        ("sbd_xlmr_baseline", "XLM-RoBERTa baseline"),
        ("sbd_xlmr_dapt",     "XLM-RoBERTa + DAPT"),
        ("sbd_xlmr_dapt_morph","XLM-RoBERTa + DAPT + Morph (novel)"),
    ]:
        full = os.path.join(MODELS_DIR, d)
        if os.path.isdir(full):
            segs[name] = ("hf", full)
    return segs


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Ayurvedic AI — Phase 2", page_icon="🌿", layout="wide")
st.title("🌿 ආයුර්වේද කෘතිම බුද්ධි පද්ධතිය — Phase 2")
st.caption("Confidence-cascading neuro-symbolic pipeline with HITL triage")

segs = list_available_segmenters()
if not segs:
    st.error("No segmenter checkpoints found in `data/models/`. Train one first via the Phase 2 Colab notebook.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Pipeline")
    seg_name = st.selectbox("Segmenter", list(segs.keys()))
    threshold = st.slider("STOP threshold τ_seg", 0.05, 0.95, 0.50, 0.05)
    window = st.slider("Safety context window k", 0, 3, 1, 1)
    hitl_tau = st.slider("HITL reliability τ_HITL", 0.0, 1.0, 0.55, 0.05,
                         help="Below this segmenter reliability the verdict is "
                              "HITL (escalate to a practitioner) even when no "
                              "toxic-ingredient issue is found.")

user_input = st.text_area(
    "Raw Sinhala Ayurvedic text (no punctuation)",
    value="වාත රෝග සඳහා නියඟලා අලයක් ගෙන හොඳින් සුද්ද කරගන්න ඉන්පසු එය ගොම දියරේ දින තුනක් ගිල්වා තබන්න පසුව වේලා කුඩු කරගන්න",
    height=140,
)

if st.button("🔍 Analyze", type="primary"):
    # ----------------------------------------------------------------------
    # Stage 1: segmentation with reliability
    # ----------------------------------------------------------------------
    kind, payload = segs[seg_name]
    if kind == "crf":
        crf = get_crf()
        seg: SegmentationResult = segment_with_crf(user_input, crf, threshold=threshold)
    else:
        predict = get_hf_token_predictor(payload)
        seg = segment_with_callable(user_input, predict, threshold=threshold, method=seg_name)

    st.markdown(f"#### 🧩 Segmented Output  (reliability {seg.reliability:.3f})")
    st.info(seg.segmented_text)

    # Per-token heatmap
    rows = []
    for w, p in zip(seg.words, seg.stop_probs):
        rows.append({"word": w, "P(STOP)": round(p, 3),
                     "decision": "STOP" if p > threshold else "O"})
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # ----------------------------------------------------------------------
    # Stage 2: cascade safety verdict
    # ----------------------------------------------------------------------
    kg = get_kg()
    if kg is None:
        st.error("Knowledge Graph CSV missing.")
    else:
        verdict = cascade_safety(seg, kg, window_size=window, hitl_min_seg_reliability=hitl_tau)
        st.markdown("### 🛡️ Safety Verdict")
        if verdict.final_status == "APPROVE":
            st.success(f"🟢 APPROVE — no issues. reliability = {verdict.seg_reliability:.3f}")
        elif verdict.final_status == "REJECT":
            st.error(f"🛑 REJECT — {verdict.issues_count} unresolved toxic mention(s).")
        else:  # HITL
            st.warning(f"⚠️ HITL — escalate to a practitioner. Reason: {verdict.hitl_reason}")
        if verdict.details:
            st.write("#### Evidence chain")
            st.dataframe(verdict.details, use_container_width=True, hide_index=True)
