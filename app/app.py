
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import streamlit as st
from pipeline import segment_text, load_knowledge_graph, analyze_safety

# --- Page Configuration ---
st.set_page_config(page_title="Ayurvedic AI Pipeline", page_icon="🌿", layout="wide")

# --- UI Header ---
st.title("🌿 ආයුර්වේද කෘතිම බුද්ධි පද්ධතිය")
st.subheader("Ayurvedic NLP & Toxicity Guardrail System")
st.markdown("---")

# --- Load Knowledge Graph ---
@st.cache_data
def get_kg():
    return load_knowledge_graph()

kg = get_kg()

# --- Sidebar Settings ---
st.sidebar.header("⚙️ පද්ධති සැකසුම් (Settings)")
st.sidebar.markdown("මෙමඟින් AI ආකෘතියේ සංවේදීතාව වෙනස් කළ හැක.")
threshold = st.sidebar.slider("Segmentation Threshold (තිත් තැබීමේ සීමාව)", 0.05, 0.50, 0.15, 0.01)
window_size = st.sidebar.slider("Context Window Size (කියවන වාක්‍ය ගණන)", 0, 3, 1, 1)

if kg is None:
    st.sidebar.error("Knowledge Graph CSV ගොනුව සොයාගත නොහැක!")

# --- Main Input Area ---
st.write("#### 📝 විරාම ලකුණු නොමැති වාක්‍ය ඛණ්ඩය (Raw Input Text)")
user_input = st.text_area("පරීක්ෂා කළ යුතු වට්ටෝරුව මෙහි ඇතුළත් කරන්න:", height=150, 
                          value="වාත රෝග සඳහා නියඟලා අලයක් ගෙන හොඳින් සුද්ද කරගන්න ඉන්පසු එය ගොම දියරේ දින තුනක් ගිල්වා තබන්න පසුව වේලා කුඩු කරගන්න")

if st.button("🔍 වට්ටෝරුව විශ්ලේෂණය කරන්න (Analyze)", type="primary", use_container_width=True):
    if user_input.strip() == "":
        st.warning("කරුණාකර පෙළක් ඇතුළත් කරන්න.")
    else:
        # Step 1: AI Segmentation
        with st.spinner("AI ආකෘතිය මඟින් වාක්‍ය ඛණ්ඩනය කරමින් පවතී..."):
            segmented_output = segment_text(user_input, threshold=threshold)
            
        st.markdown("### 🧩 ඛණ්ඩනය කළ වාක්‍ය (Segmented Output)")
        st.info(segmented_output)
        
        # Step 2: Toxicity Guardrail
        st.markdown("### 🛡️ වෛද්‍ය ආරක්ෂක පරීක්ෂාව (Safety Guardrail Report)")
        
        if kg:
            with st.spinner("Knowledge Graph හරහා විෂ ඖෂධ පරීක්ෂා කරමින් පවතී..."):
                report = analyze_safety(segmented_output, kg, window_size=window_size)
            
            # Display Final Status
            if report["final_status"] == "APPROVED":
                st.success(f"🟢 **අනුමතයි (APPROVED)** - වට්ටෝරුව භාවිතයට ආරක්ෂිතයි! (අනතුරුදායක දෝෂ: {report['issues_count']})")
            else:
                st.error(f"🛑 **ප්‍රතික්ෂේපිතයි (REJECTED)** - වට්ටෝරුවේ ජීවිත අවදානමක් ඇත! (අනතුරුදායක දෝෂ: {report['issues_count']})")
                
            # Display Detailed Report
            st.write("#### 📋 සවිස්තරාත්මක විශ්ලේෂණය (Detailed Analysis):")
            if not report["details"]:
                st.write("කිසිදු විෂ සහිත ඖෂධයක් හමු නොවීය.")
            else:
                for item in report["details"]:
                    if item["status"] == "PASS":
                        st.success(item["message"])
                    else:
                        st.error(item["message"])