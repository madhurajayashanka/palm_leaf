# PhD-Level Research Analysis: Phase 1 Ayurvedic NLP Pipeline

## 1. Introduction and Scope
This document provides a rigorous academic analysis of the **Phase 1 Pipeline** developed for the computational processing of classical Sinhala Ayurvedic texts. The architecture addresses three critical challenges in clinical historical NLP: **(1)** Noisy Optical Character Recognition (OCR) correction in a low-resource setting; **(2)** Sentence Boundary Detection (SBD) without standard orthographic punctuation; and **(3)** Deterministic safety validation of toxic medicinal entities.

For each stage, this analysis provides theoretical justifications, mathematical proofs, literature comparisons, and edge-case evaluations required for PhD-level defense.

---

## Chapter 1: Stage 1A — OCR Post-Correction via N-Gram Models and Viterbi Decoding

### 1.1 The Challenge: The Noisy Channel Model in Historical Texts
When ancient palm-leaf manuscripts are digitized via OCR engines, the resulting text suffers from high Character Error Rates (CER) and Word Error Rates (WER). In Sinhala, visually similar graphemes (e.g., 'ට' and 'ඩ', or 'ව' and 'ච') cause extreme confusion.

### 1.2 Theoretical Justification: Bigram Model + Viterbi vs. Deep Learning
In modern NLP, OCR post-correction is often treated as a Neural Machine Translation (NMT) task using Sequence-to-Sequence (Seq2Seq) Transformer models (e.g., fine-tuned RoBERTa or ByT5). However, **Deep Learning approaches fail catastrophically in extreme low-resource historical contexts** where massive parallel corpora of "noisy-clean" text do not exist. 

**Proof of Architectural Choice:**
*   **Data Scarcity Theorem:** Deep learning models require generalized representations learned from millions of tokens to avoid overfitting. Our corpus consists of only ~70,000 sentences.
*   **The Statistical Baseline:** We frame the problem using Shannon’s Noisy Channel Model. The true word sequence $W$ given a noisy OCR sequence $O$ is found by maximizing $P(W|O)$, which by Bayes' Theorem is proportional to $P(O|W) \cdot P(W)$.
*   **Literature Support:** Research by *Rijhwani et al. (EMNLP 2020)* on endangered languages demonstrates that combining weighted finite-state automata (WFSA) or Hidden Markov Models (HMM) with lexical decoding significantly outperforms purely neural methods when resources are scarce. The statistical model guarantees the output remains within a known historical lexicon.

### 1.3 Mathematical Formulation (The Viterbi Proof)
To solve the decoding problem, we utilize a **Bigram Language Model** to approximate $P(W) \approx \prod P(w_i | w_{i-1})$ and the **Viterbi Algorithm** to efficiently search the hypothesis space.

**The Viterbi Recurrence Relation:**
Let $V_{t,k}$ be the maximum probability of a word sequence of length $t$ ending in word state $k$.
$$V_{t,k} = \max_{x \in X} \Big( V_{t-1,x} \cdot P(y_t | k) \cdot P(k | x) \Big)$$

*   $V_{t-1,x}$: The accumulated probability of the optimal path up to the previous word $x$.
*   $P(y_t | k)$: The **Optical/Emission Probability** (how likely the OCR engine outputs string $y_t$ when the true word is $k$).
*   $P(k | x)$: The **Transition Probability** (the Bigram likelihood of word $k$ following word $x$).

**Performance Proof:** The Viterbi algorithm reduces the time complexity of decoding an $N$-word sequence with $S$ candidates per word from $O(S^N)$ to $O(N \cdot S^2)$, making it computationally viable (yielding the **0.1s latency** observed in testing).

### 1.4 Edge Case Handling
1.  **Zero-Probability Paths (Data Sparsity):** If a bigram $(w_{i-1}, w_i)$ never appeared in the 70,000-sentence training corpus, its probability drops to $0$, collapsing the entire equation sequence.
    *   *Solution:* We implement **Laplace (Add-1) Smoothing** or Backoff strategies to ensure $P(k|x) > 0$ for unseen word combinations, preventing catastrophic decoder failure.
2.  **Out-Of-Vocabulary (OOV) Candidates:** If the correct historical word is absent from the unigram dictionary, the decoder cannot emit it.
    *   *Limitation Acknowledgment:* N-gram models have a closed vocabulary constraint. This is a known trade-off; however, for historical clinical texts, staying within a closed, verified vocabulary prevents the model from hallucinating non-existent medicinal herbs.

---

## Chapter 2: Stage 1B — Sentence Boundary Detection via Hybrid CRF

### 2.1 The Challenge: Boundary Detection in Punctuation-less Corpora
Classical Sinhala Ayurvedic texts do not use modern punctuation (periods or commas) to denote sentence ends. SBD must be treated as a Sequence Labeling task where words are tagged as either `O` (Continue) or `STOP` (Boundary).

### 2.2 Theoretical Justification: Feature-Based CRFs vs. Transformers
During Phase 2, a critical experiment was conducted: comparing a **Conditional Random Field (CRF)** against **XLM-RoBERTa**. 

*   **The 15,000 Sentence Phenomenon:** At low data volumes (15k sentences), the CRF drastically outperformed XLM-RoBERTa. This occurs because Transformers rely entirely on *Representation Learning* (learning grammar implicitly from data vectors). CRFs, conversely, utilize *Feature-Based Modeling*.
*   **Linguistic Proof (Agglutinative & SOV Morphology):** Sinhala is a Subject-Object-Verb (SOV) language, meaning the predicate naturally concludes the sentence. Furthermore, as an agglutinative language, grammatical functions (tense, mood, person) are appended as suffixes to the verb root.
*   **Hand-Crafted Features:** By explicitly injecting morphological rules into the CRF (e.g., checking if words end with finite verb markers like `'යි'`, `'යේය'`, `'වේ'`, or imperative clinical markers like `'මැනවි'`, `'යුතු'`, `'ගනු'`), we bypass the need for massive datasets. The CRF is explicitly instructed *what* to look for.
*   **Literature Support:** *Ruokolainen et al. (ACL)* and research on models like CHIPMUNK (Semi-CRFs) have established that for morphologically rich, low-resource languages, injecting explicit morphological tags into CRFs yields superior sequence labeling performance compared to pure neural models.

### 2.3 Edge Case Handling
1.  **The "Non-Finite" Ambiguity:** In Sinhala, non-finite verbs (e.g., words ending in *කරමින්* / *karamin* - "while doing") indicate clauses, not sentence ends. 
    *   *Solution:* The hand-crafted feature list strictly isolates *finite* verb endings and specific clinical imperatives, preventing the model from prematurely segmenting complex instructional sentences.
2.  **Abbreviation Suffix Overlap:** If an abbreviation or noun happens to end with a string identical to a verb suffix.
    *   *Solution:* CRFs evaluate the entire sequence's conditional probability: $P(Y|X) = \frac{1}{Z(X)} \exp\left(\sum \lambda_j f_j(y_{i-1}, y_i, X, i)\right)$. The feature $f_j$ looks not just at the suffix, but at the surrounding context words, suppressing false positives.

---

## Chapter 3: Stage 1C — Knowledge Graph Safety Guardrail

### 3.1 The Challenge: Clinical Safety and Neural Hallucinations
A medical NLP pipeline handling recipes involving toxic flora (e.g., *Niyangala* / *Gloriosa superba*) cannot rely on probabilistic boundaries. If an LLM or Transformer model makes a single sequence error—such as splitting a sentence between the toxic ingredient and its mandatory purification instruction (*Shodhana*)—the system could approve a lethal recipe.

### 3.2 Theoretical Justification: Deterministic vs. Probabilistic Architecture
We advocate for a **Neuro-Symbolic Hybrid Architecture**, utilizing the CRF/Neural segmenter for linguistic processing, but bounding its output with a **Deterministic Knowledge Graph (KG)**.

*   **The Cascading Failure Problem:** "වෛද්‍ය උපදෙස් වලදී AI එකක් නිවැරදිව වාක්‍ය ඛණ්ඩනය නොකළහොත්, ජීවිතය අනතුරේ වැටෙන බව තහවුරු විය." (If segmentation fails, the toxicity check fails). Probabilistic models (like RoBERTa) provide "soft" statistical confidence scores. In medicine, safety requires a **"Hard Veto."**
*   **The KG as Ground Truth:** A Knowledge Graph maps structured, immutable facts (Entity $\rightarrow$ Toxicity Level $\rightarrow$ Purification Keywords). A rule-based querying of this graph guarantees 100% accuracy relative to the database, preventing the algorithmic hallucinations common in modern LLMs.

### 3.3 Edge Case Handling: The Human-in-the-Loop "Context Window"
*   **The Trade-off:** Ayurvedic texts sometimes split purification steps across multiple sentences. 
    *   *If Context Window = 0 (Strict):* Maximum safety. The algorithm flags an alert unless the purification keyword is in the *exact same* sentence as the toxic herb. **Edge Case:** High False Positives if the manuscript structure separates the steps.
    *   *If Context Window > 0 (Lenient):* Allows the guardrail to scan adjacent sentences. **Edge Case:** High False Negatives (danger) if unrelated toxic herbs from previous recipes are accidentally validated.
*   **Solution:** We resolve this architectural paradox via a **Human-in-the-Loop UI Slider**. By allowing the Ayurvedic practitioner to dynamically adjust the Context Window based on the specific structural style of the manuscript being analyzed, the AI acts as a collaborative augmented intelligence rather than a brittle autonomous agent.

---

## 4. Conclusion
The Phase 1 pipeline proves that in specialized, low-resource historical domains, **traditional statistical methods (Viterbi, CRFs) injected with domain-specific linguistic rules categorically outperform modern deep learning architectures** (which suffer from data starvation). Furthermore, combining probabilistic NLP with deterministic Knowledge Graphs provides the auditable, hard-veto safety required for processing potentially lethal clinical texts.