import json

def load_language_model(file_path):
    """Loads the trained bigram probabilities."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            print(f"Successfully loaded Language Model from '{file_path}'")
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Please run the Bigram Generator first.")
        return {}

def viterbi_decode(ocr_input, lm_probabilities, alpha=0.5, beta=0.5):
    """
    Decodes the best sentence path using OCR confidence and LM probabilities.
    alpha: Weight for OCR Confidence (Acoustic/Scanner Model)
    beta:  Weight for Language Model Probability (Linguistic Context)
    """
    V = [{}]
    
    # Step 1: Initialize the first word probabilities
    for candidate in ocr_input[0]['candidates']:
        word = candidate['word']
        V[0][word] = {"score": candidate['confidence'], "prev": None}

    # Step 2: Run Viterbi for the rest of the sequence
    for t in range(1, len(ocr_input)):
        V.append({})
        for candidate in ocr_input[t]['candidates']:
            current_word = candidate['word']
            ocr_conf = candidate['confidence']
            
            max_tr_score = -1
            best_prev_word = None
            
            # Check all possible words from the previous step
            for prev_word in V[t-1].keys():
                
                # NLP Smoothing: If the pair has never been seen in your training text, 
                # give it a tiny non-zero probability so the math doesn't collapse to absolute zero.
                lm_prob = 0.0001 
                if prev_word in lm_probabilities and current_word in lm_probabilities[prev_word]:
                    lm_prob = lm_probabilities[prev_word][current_word]
                
                # The Core Equation: Score = Previous Path Score * (Weighted OCR + Weighted LM)
                path_score = V[t-1][prev_word]["score"] * ((alpha * ocr_conf) + (beta * lm_prob))
                
                if path_score > max_tr_score:
                    max_tr_score = path_score
                    best_prev_word = prev_word
                    
            V[t][current_word] = {"score": max_tr_score, "prev": best_prev_word}

    # Step 3: Backtrack to find the best path
    opt = []
    max_final_score = -1
    best_last_word = None
    
    # Find the highest scoring word at the very end of the sentence
    for word, data in V[-1].items():
        if data["score"] > max_final_score:
            max_final_score = data["score"]
            best_last_word = word
            
    opt.append(best_last_word)
    previous = best_last_word
    
    # Follow the "prev" pointers backwards to the beginning
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    return " ".join(opt)

# --- Execution ---

# Here is a snippet of your OCR input array
ocr_data = [
  { "candidates": [ { "word": "සහ", "confidence": 0.98 }, { "word": "මහා", "confidence": 0.12 }, { "word": "ගහ", "confidence": 0.05 } ] },
  { "candidates": [ { "word": "සමේ", "confidence": 0.95 }, { "word": "ගමේ", "confidence": 0.15 }, { "word": "කමේ", "confidence": 0.08 } ] },
  { "candidates": [ { "word": "රෝග", "confidence": 0.92 }, { "word": "බෝග", "confidence": 0.18 }, { "word": "යෝග", "confidence": 0.04 } ] },
  { "candidates": [ { "word": "ඇති", "confidence": 0.96 }, { "word": "නැති", "confidence": 0.10 }, { "word": "අති", "confidence": 0.05 } ] },
  { "candidates": [ { "word": "තැන", "confidence": 0.009 }, { "word": "බීම", "confidence": 0.22 }, { "word": "කීම", "confidence": 0.11 } ] }
]

# Load your custom trained model
lm_model = load_language_model('bigram_probabilities.json')

if lm_model:
    print("\nRunning Viterbi Decoding...")
    
    # Tweak alpha and beta to test! 
    # alpha=0.6 means you trust the scanner slightly more than the grammar rules.
    best_sentence = viterbi_decode(ocr_data, lm_model, alpha=0.6, beta=0.4)
    
    print("\nFinal Decoded Ayurvedic Text:")
    print(">>", best_sentence)