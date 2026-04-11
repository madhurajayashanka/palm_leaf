import json
import math
from config import VITERBI_ALPHA, VITERBI_BETA, VITERBI_SMOOTHING, normalize_sinhala, DATA_DIR
import os

def load_language_model(file_path):
    """Loads the trained bigram probabilities."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def viterbi_decode(ocr_input, lm_probabilities, alpha=None, beta=None, smoothing=None):
    """
    Decodes the best sentence path using OCR confidence and LM probabilities.
    Uses LOG-SPACE arithmetic to prevent underflow on long sequences.
    
    alpha: Weight for OCR Confidence (Scanner Model)
    beta:  Weight for Language Model Probability (Linguistic Context)
    smoothing: Probability assigned to unseen bigrams
    """
    if alpha is None:
        alpha = VITERBI_ALPHA
    if beta is None:
        beta = VITERBI_BETA
    if smoothing is None:
        smoothing = VITERBI_SMOOTHING

    if not ocr_input:
        return ""
    
    # Handle single-position input
    if len(ocr_input) == 1:
        if not ocr_input[0].get('candidates'):
            return ""
        best = max(ocr_input[0]['candidates'], key=lambda c: c['confidence'])
        return best['word']

    V = [{}]
    
    # Step 1: Initialize the first word log-probabilities
    for candidate in ocr_input[0]['candidates']:
        word = normalize_sinhala(candidate['word'])
        conf = max(candidate['confidence'], 1e-10)  # Floor to avoid log(0)
        V[0][word] = {"score": math.log(conf), "prev": None}

    # Step 2: Run Viterbi for the rest of the sequence (log-space)
    for t in range(1, len(ocr_input)):
        V.append({})
        if not ocr_input[t].get('candidates'):
            continue
        for candidate in ocr_input[t]['candidates']:
            current_word = normalize_sinhala(candidate['word'])
            ocr_conf = max(candidate['confidence'], 1e-10)
            
            max_log_score = -float('inf')
            best_prev_word = None
            
            for prev_word in V[t-1].keys():
                # Look up bigram probability with smoothing fallback
                lm_prob = smoothing
                if prev_word in lm_probabilities and current_word in lm_probabilities[prev_word]:
                    lm_prob = lm_probabilities[prev_word][current_word]
                
                # Log-linear interpolation
                log_score = V[t-1][prev_word]["score"] + alpha * math.log(ocr_conf) + beta * math.log(lm_prob)
                
                if log_score > max_log_score:
                    max_log_score = log_score
                    best_prev_word = prev_word
                    
            V[t][current_word] = {"score": max_log_score, "prev": best_prev_word}

    # Step 3: Backtrack to find the best path
    if not V[-1]:
        return ""
    
    best_last_word = max(V[-1], key=lambda w: V[-1][w]["score"])
    
    opt = [best_last_word]
    previous = best_last_word
    
    for t in range(len(V) - 2, -1, -1):
        prev = V[t + 1][previous]["prev"]
        if prev is None:
            break
        opt.insert(0, prev)
        previous = prev

    return " ".join(opt)

# --- Execution ---

# Sample OCR input array (simulated scanner output)
ocr_data = [
  { "candidates": [ { "word": "සහ", "confidence": 0.98 }, { "word": "මහා", "confidence": 0.12 }, { "word": "ගහ", "confidence": 0.05 } ] },
  { "candidates": [ { "word": "සමේ", "confidence": 0.95 }, { "word": "ගමේ", "confidence": 0.15 }, { "word": "කමේ", "confidence": 0.08 } ] },
  { "candidates": [ { "word": "රෝග", "confidence": 0.92 }, { "word": "බෝග", "confidence": 0.18 }, { "word": "යෝග", "confidence": 0.04 } ] },
  { "candidates": [ { "word": "ඇති", "confidence": 0.96 }, { "word": "නැති", "confidence": 0.10 }, { "word": "අති", "confidence": 0.05 } ] },
  { "candidates": [ { "word": "තැන", "confidence": 0.009 }, { "word": "බීම", "confidence": 0.22 }, { "word": "කීම", "confidence": 0.11 } ] }
]

if __name__ == "__main__":
    lm_model = load_language_model(os.path.join(DATA_DIR, 'bigram_probabilities.json'))

    if lm_model:
        print("Running Viterbi Decoding (log-space)...")
        best_sentence = viterbi_decode(ocr_data, lm_model)
        print(f"Final Decoded Ayurvedic Text: >> {best_sentence}")