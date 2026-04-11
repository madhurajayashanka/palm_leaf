import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import streamlit as st
import time
import json
from pipeline import segment_text, load_knowledge_graph, analyze_safety
from viterbi_decoder import viterbi_decode, load_language_model
from config import DATA_DIR

# --- Page Configuration ---
st.set_page_config(page_title="Ayurvedic AI Pipeline", page_icon="🌿", layout="wide")

# --- UI Header ---
st.title("🌿 ආයුර්වේද කෘතිම බුද්ධි පද්ධතිය")
st.subheader("Ayurvedic Neuro-Symbolic NLP Pipeline — Full Demo")
st.markdown("---")

# --- Load Resources ---
@st.cache_data
def get_kg():
    return load_knowledge_graph()

@st.cache_data
def get_lm():
    return load_language_model(os.path.join(DATA_DIR, "bigram_probabilities.json"))

kg = get_kg()
lm = get_lm()

# --- Sidebar Settings ---
st.sidebar.header("⚙️ පද්ධති සැකසුම් (Settings)")
threshold = st.sidebar.slider("Segmentation Threshold (තිත් තැබීමේ සීමාව)", 0.05, 0.50, 0.15, 0.01)
window_size = st.sidebar.slider("Context Window Size (කියවන වාක්‍ය ගණන)", 0, 3, 1, 1)
alpha = st.sidebar.slider("OCR Confidence Weight (α)", 0.0, 1.0, 0.6, 0.05)
beta_val = st.sidebar.slider("Language Model Weight (β)", 0.0, 1.0, 0.4, 0.05)

if kg is None:
    st.sidebar.error("Knowledge Graph CSV ගොනුව සොයාගත නොහැක!")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs([
    "🔬 Full Pipeline (සම්පූර්ණ නල මාර්ගය)",
    "📊 OCR Correction Demo (OCR නිවැරදි කිරීම)",
    "📖 Architecture (ගෘහනිර්මාණ ශිල්පය)"
])

# ==========================================
# TAB 1: FULL PIPELINE
# ==========================================
with tab1:
    st.write("#### 📝 විරාම ලකුණු නොමැති වාක්‍ය ඛණ්ඩය (Raw Input Text)")
    user_input = st.text_area(
        "පරීක්ෂා කළ යුතු වට්ටෝරුව මෙහි ඇතුළත් කරන්න:",
        height=150,
        value="වාත රෝග සඳහා නියඟලා අලයක් ගෙන හොඳින් සුද්ද කරගන්න ඉන්පසු එය ගොම දියරේ දින තුනක් ගිල්වා තබන්න පසුව වේලා කුඩු කරගන්න"
    )

    if user_input.strip() == "":
        st.warning("කරුණාකර පෙළක් ඇතුළත් කරන්න.")
    else:
        # --- Stage 1B: Segmentation ---
        st.markdown("### 🧩 Stage 1B: CRF Sentence Segmentation")
        t_start = time.time()
        segmented_output = segment_text(user_input, threshold=threshold)
        t_seg = time.time() - t_start

        if segmented_output.startswith("Error:"):
            st.error(segmented_output)
            st.info("💡 **Fix:** Run the Phase 1 notebook to train and export `ayurvedic_segmenter.pkl`")
        else:
            st.caption(f"⏱️ Segmentation latency: {t_seg*1000:.1f} ms")
            
            st.info("💡 **Human-in-the-Loop:** You can manually edit the sentence boundaries (periods) below before the Safety Guardrail runs. The analysis will instantly update.")
            
            user_edited_segmentation = st.text_area(
                "AI Output (මානව සංස්කරණය සඳහා):",
                value=segmented_output,
                height=100
            )

            # Count segments
            sentences = [s.strip() for s in user_edited_segmentation.split('.') if s.strip()]
            st.write(f"**Sentences detected:** {len(sentences)}")

            # --- Stage 1C: Safety Guardrail ---
            st.markdown("### 🛡️ Stage 1C: Knowledge Graph Safety Guardrail")

            if kg:
                t_start = time.time()
                report = analyze_safety(user_edited_segmentation, kg, window_size=window_size)
                t_safety = time.time() - t_start

                if report["final_status"] == "APPROVED":
                    st.success(f"🟢 **අනුමතයි (APPROVED)** — Issues: {report['issues_count']}")
                else:
                    st.error(f"🛑 **ප්‍රතික්ෂේපිතයි (REJECTED)** — Issues: {report['issues_count']}")

                st.caption(f"⏱️ Safety check latency: {t_safety*1000:.1f} ms")

                # Detailed report
                st.write("#### 📋 Detailed Analysis:")
                if not report["details"]:
                    st.write("කිසිදු විෂ සහිත ඖෂධයක් හමු නොවීය.")
                else:
                    for item in report["details"]:
                        if item["status"] == "PASS":
                            st.success(item["message"])
                        else:
                            st.error(item["message"])

                # --- Cascading Failure Demo ---
                st.markdown("---")
                st.markdown("### ⚠️ Cascading Failure Demonstration")
                st.caption(
                    "Adjusting the segmentation threshold changes where sentence breaks land, "
                    "which can cause the safety guardrail to give different verdicts for the same input."
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Strict segmentation (θ=0.10)**")
                    seg_strict = segment_text(user_input, threshold=0.10)
                    if not seg_strict.startswith("Error:"):
                        st.code(seg_strict, language=None)
                        r = analyze_safety(seg_strict, kg, window_size=window_size)
                        if r["final_status"] == "APPROVED":
                            st.success(f"APPROVED ({r['issues_count']} issues)")
                        else:
                            st.error(f"REJECTED ({r['issues_count']} issues)")

                with col2:
                    st.write("**Relaxed segmentation (θ=0.30)**")
                    seg_relaxed = segment_text(user_input, threshold=0.30)
                    if not seg_relaxed.startswith("Error:"):
                        st.code(seg_relaxed, language=None)
                        r = analyze_safety(seg_relaxed, kg, window_size=window_size)
                        if r["final_status"] == "APPROVED":
                            st.success(f"APPROVED ({r['issues_count']} issues)")
                        else:
                            st.error(f"REJECTED ({r['issues_count']} issues)")
            else:
                st.error("Knowledge Graph not loaded.")

# ==========================================
# TAB 2: OCR CORRECTION DEMO
# ==========================================
with tab2:
    st.markdown("### 🔤 Stage 1A: Bigram LM + Viterbi OCR Post-Correction")
    st.caption(
        "Simulates OCR output where the scanner gives multiple candidate words per position with confidence scores. "
        "The Viterbi decoder finds the best path by combining OCR confidence (α) and language model probability (β)."
    )

    # Predefined demo OCR data
    demo_ocr = [
        {"candidates": [{"word": "සහ", "confidence": 0.98}, {"word": "මහා", "confidence": 0.12}, {"word": "ගහ", "confidence": 0.05}]},
        {"candidates": [{"word": "සමේ", "confidence": 0.95}, {"word": "ගමේ", "confidence": 0.15}, {"word": "කමේ", "confidence": 0.08}]},
        {"candidates": [{"word": "රෝග", "confidence": 0.92}, {"word": "බෝග", "confidence": 0.18}, {"word": "යෝග", "confidence": 0.04}]},
        {"candidates": [{"word": "ඇති", "confidence": 0.96}, {"word": "නැති", "confidence": 0.10}, {"word": "අති", "confidence": 0.05}]},
        {"candidates": [{"word": "තැන", "confidence": 0.009}, {"word": "බීම", "confidence": 0.22}, {"word": "කීම", "confidence": 0.11}]},
    ]

    # Show OCR candidate lattice
    st.write("**OCR Candidate Lattice:**")
    for pos_idx, position in enumerate(demo_ocr):
        cols = st.columns(len(position["candidates"]) + 1)
        cols[0].write(f"**Position {pos_idx + 1}**")
        for c_idx, c in enumerate(position["candidates"]):
            conf_color = "🟢" if c["confidence"] > 0.7 else "🟡" if c["confidence"] > 0.2 else "🔴"
            cols[c_idx + 1].write(f"{conf_color} `{c['word']}` ({c['confidence']:.3f})")

    if st.button("🔧 Run Viterbi Decoding", type="primary"):
        if lm:
            t_start = time.time()
            decoded = viterbi_decode(demo_ocr, lm, alpha=alpha, beta=beta_val)
            t_ocr = time.time() - t_start

            st.markdown("**Decoded output:**")
            st.success(f"📝 {decoded}")
            st.caption(f"⏱️ Decoding latency: {t_ocr*1000:.1f} ms")

            # Show what "greedy OCR" would have picked (highest confidence per position)
            greedy = " ".join(
                max(pos["candidates"], key=lambda c: c["confidence"])["word"]
                for pos in demo_ocr
            )
            st.write(f"**Greedy OCR baseline (just pick highest confidence):** `{greedy}`")
            st.write(f"**Viterbi + LM correction:** `{decoded}`")

            if greedy != decoded:
                st.info("✨ The language model corrected the output vs. greedy OCR selection!")
            else:
                st.caption("In this case, greedy and Viterbi agree.")
        else:
            st.error("Language model not loaded. Ensure bigram_probabilities.json exists in data/.")

    st.markdown("---")
    st.markdown("#### α/β Sensitivity Analysis")
    st.caption("Vary α (OCR weight) and β (LM weight) to see how the trade-off affects decoding.")

    if lm:
        results = []
        for a_test in [0.3, 0.5, 0.6, 0.7, 0.9]:
            b_test = 1.0 - a_test
            decoded = viterbi_decode(demo_ocr, lm, alpha=a_test, beta=b_test)
            results.append({"α (OCR)": a_test, "β (LM)": b_test, "Decoded Text": decoded})

        st.table(results)

# ==========================================
# TAB 3: ARCHITECTURE OVERVIEW
# ==========================================
with tab3:
    st.markdown("### System Architecture")
    st.markdown("""
    ```
    ┌──────────────────┐     ┌──────────────────────┐     ┌─────────────────────────┐     ┌──────────────────────┐
    │  OCR Scanner      │────▶│ Bigram LM + Viterbi  │────▶│ CRF/RoBERTa Segmenter   │────▶│ KG Safety Guardrail  │
    │  (External)       │     │ Stage 1A              │     │ Stage 1B / Phase 2      │     │ Stage 1C              │
    └──────────────────┘     └──────────────────────┘     └─────────────────────────┘     └──────────────────────┘
         Noisy OCR                 Corrected text              Segmented sentences           APPROVED / REJECTED
    ```
    """)

    st.markdown("### Design Principles")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🔒 Hard Veto Safety**
        The KG's REJECTED verdict is absolute — neural confidence scores cannot override it.

        **🔗 Neuro-Symbolic Complementarity**
        Neural/statistical models handle linguistic ambiguity; deterministic KG handles safety decisions.
        """)
    with col2:
        st.markdown("""
        **👨‍⚕️ Human-in-the-Loop**
        Context window slider lets practitioners modulate the safety-context trade-off.

        **📉 Graceful Degradation**
        Each stage fails safely — missing models produce clear error messages, not silent failures.
        """)

    st.markdown("### Knowledge Graph Statistics")
    if kg:
        total_entities = len(kg)
        high_tox = sum(1 for d in kg.values() if "high" in d["toxicity"].lower())
        med_tox = sum(1 for d in kg.values() if "medium" in d["toxicity"].lower())
        total_aliases = sum(len(d["aliases"]) for d in kg.values())
        total_keywords = sum(len(d["shodhana_keywords"]) for d in kg.values())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Toxic Entities", total_entities)
        c2.metric("High Toxicity", high_tox)
        c3.metric("Medium Toxicity", med_tox)
        c4.metric("Total Aliases", total_aliases)
        st.write(f"**Total purification keywords:** {total_keywords}")
    else:
        st.warning("KG not loaded.")

    st.markdown("### Key Research Finding")
    st.info(
        "**CRF with domain-specific Sinhala morphological features outperforms XLM-RoBERTa (278M params) "
        "for sentence boundary detection when training data ≤ 15,000 sentences.** "
        "The transformer requires approximately 4.7× more data to reach comparable performance."
    )
