# Research Justifications & Thesis Notes: Ayurvedic NLP Pipeline

This document provides academic, linguistic, and architectural justifications required for a final-year research thesis. It bridges the gap between the implemented code and the theoretical proofs needed for academic defense.

---

## 1. Linguistic Justification for Sinhala Sentence Boundary Detection (SBD)

### 1.1 The Challenge of Sinhala NLP in Classical Texts
Unlike English, where punctuation (periods, question marks) strictly dictates sentence boundaries, ancient and classical Sinhala texts (like Ayurvedic palm-leaf manuscripts) often lack standard punctuation. Therefore, Sentence Boundary Detection (SBD) must rely on syntactic and morphological cues.

### 1.2 Justification for Chosen Morphological Features (Verb Endings)
In the Phase 2 Segmenter, specific suffixes and words were hardcoded as features for the Conditional Random Field (CRF) model: `['යි', 'ස්', 'යුතු', 'යේය', 'වේ', 'මැනවි', 'ගනු', 'පෙර', 'පසු', 'කරයි', 'න්න', 'ගන්න', 'තබන්න']`.

**Academic Proof & Reasoning:**
1.  **Subject-Object-Verb (SOV) Structure:** Sinhala is strictly an SOV language. Therefore, the predicate (typically a finite verb) naturally occurs at the absolute end of a sentence.
2.  **Finite Verbs vs. Non-Finite Verbs:** In Sinhala grammar, non-finite verbs (e.g., *කරමින්* / *karamin* - "while doing") indicate clause boundaries (continuations), whereas **finite verbs** conclude a thought. The suffixes `'යි'` (yi), `'යේය'` (yeya), and `'වේ'` (we) are classical finite verb markers indicating the end of a declarative statement.
3.  **Imperative Mood in Instructional Texts:** Ayurvedic texts are inherently instructional (recipes and treatments). Therefore, sentences frequently end with imperative verbs or prescriptive terms. Words like `'යුතු'` (yuthu - "must"), `'මැනවි'` (manawi - "it is good to / one should"), `'ගනු'` (ganu - "take"), `'න්න'` (nna - informal imperative ending like *කරන්න*), and `'තබන්න'` (thabanna - "keep") are syntactic boundary markers in instructional Sinhala.
4.  **Temporal Postpositions:** Words like `'පෙර'` (pera - "before") and `'පසු'` (pasu - "after") often conclude dependent temporal clauses that function as distinct structural segments in medicinal preparation steps.

*Reference Anchor:* These rules align with established Sinhala NLP research which states that morphologically rich languages require SBD systems to leverage morphological analyzers (identifying verbs and agglutinative suffixes) rather than relying solely on punctuation.

---

## 2. Phase 1: Baseline N-Gram (Bigram) Model

### 2.1 Why a Bigram Model? (Proof of Concept - v1.0)
Before applying Deep Learning, a statistical **N-gram Language Model** combined with a Viterbi Decoder was implemented to process noisy OCR output. 
*   **Reasoning:** OCR engines scanning ancient palm leaves produce high rates of character-level confusion (e.g., mistaking 'ට' for 'ඩ'). A Bigram model probabilistically calculates the likelihood of word $w_2$ following $w_1$. It is mathematically lightweight, highly robust to random noise, and does not require extensive computational resources.
*   **Performance:** Achieved a fast **0.1s latency** with a **75% accuracy** baseline on Ayurvedic texts. This validates the pipeline's core logic before scaling to heavier neural networks.

---

## 3. Phase 2: Weak Supervision & The CRF vs. RoBERTa Dilemma

### 3.1 Weak Supervision for Data Generation
To train a supervised model (CRF or Transformer), large amounts of labeled data are required. Manual labeling of ancient medical texts is impossible due to a lack of domain experts. 
*   **Solution:** We developed a "Weak Supervision" script to automatically generate `train_labeled.tsv`. By applying our linguistic rules (Section 1.2) to raw text, we programmatically tagged words with `O` (continue) or `STOP` (boundary), instantly creating a massive dataset for model training.

### 3.2 The Low-Resource Phenomenon: CRF > XLM-RoBERTa (at 15,000 sentences)
A critical experiment was conducted comparing a **Hybrid Conditional Random Field (CRF)** against a state-of-the-art **XLM-RoBERTa Transformer**.

*   **The 15k Data Threshold:** When trained on a limited dataset of 15,000 sentences, the CRF significantly outperformed RoBERTa.
*   **Academic Justification:** Transformers (like RoBERTa) are "data-hungry." They learn representations from scratch and suffer from **data scarcity** in low-resource languages like Sinhala. The CRF succeeded because we injected **Domain Knowledge** (hand-crafted morphological rules like 'න්න', 'වස්'). The CRF mathematically enforces sequence consistency based on these rules, compensating for the lack of training data.
*   **The 70k Data Threshold:** Only when the dataset was expanded to 70,000 sentences (using an 80/20 train/test split) did XLM-RoBERTa begin to overcome its data scarcity and produce reliable segmentations (achieving **92% accuracy** at a slower **1.2s latency**).

> **Core Research Conclusion:** 
> "සිංහල වැනි අවම දත්ත ඇති (Low-resource) භාෂාවකදී, ලෝකයේ දියුණුම Deep Learning මොඩලයකට වඩා, අප අතින් භාෂාමය නීති (Linguistic rules) ඇතුළත් කළ කුඩා CRF මොඩලය ඉතා සාර්ථකයි." 
> *(In low-resource languages like Sinhala, a small CRF model injected with linguistic rules is far more successful than the world's most advanced Deep Learning models when data is limited.)*

---

## 4. Architectural Safety & Toxicity Guardrails

### 4.1 The Threat of Cascading Failures
The most critical finding of this research is the dependency chain between the NLP architecture and patient safety.
*   **Reasoning:** If the AI incorrectly segments a sentence (e.g., merging a toxic ingredient with a purification step from the *next* sentence), the **Toxicity Guardrail** will falsely classify a lethal recipe as safe. 
*   **Conclusion:** *"වෛද්‍ය උපදෙස් වලදී AI එකක් නිවැරදිව වාක්‍ය ඛණ්ඩනය (Segmentation) නොකළහොත්, ඉන්පසුව එන විෂ පරීක්ෂා කිරීමේ පද්ධතියද (Toxicity Check) අසාර්ථක වී රෝගියාගේ ජීවිතය අනතුරේ වැටෙන බව RoBERTa ආකෘතියෙන් තහවුරු විය."* (RoBERTa models demonstrated that if AI fails to accurately segment sentences in medical instructions, the subsequent Toxicity Check fails, endangering patient lives.)

### 4.2 Human-in-the-Loop Architecture (Context Window Slider)
To mitigate the risk of cascading failures, a dynamic `Context Window Size` slider was introduced in the UI.
*   **Trade-off Justification:** 
    *   `Window = 0` (Strict Safety): The algorithm only looks for purification keywords within the exact segmented sentence. This guarantees maximum safety but may flag false positives if the Ayurvedic text structurally splits purification steps across multiple lines.
    *   `Window = 1` (Contextual): Allows the algorithm to look one sentence backward/forward.
*   **Architectural Defense:** "We identified a critical trade-off between strict safety and contextual understanding. To solve this without risking patient lives, our architecture integrates a **'Human-in-the-loop'** UI approach, allowing Ayurvedic practitioners to dynamically adjust the context window based on the structural complexity of the specific ancient text."
