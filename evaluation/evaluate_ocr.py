import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import random
from viterbi_decoder import viterbi_decode, load_language_model
from config import DATA_DIR

def simulate_ocr_noise(text, error_rate=0.15):
    # visually similar Sinhala characters
    confusions = {
        'ට': ['ඩ', 'ව'], 'ඩ': ['ට', 'ඬ'], 'ව': ['ට', 'ච'], 'ච': ['ව'],
        'ප': ['ජ'], 'ම': ['ඹ', 'ස'], 'ස': ['ම'], 'ත': ['න'], 'න': ['ත'],
        'ය': ['යු', 'ස'], 'ග': ['ඟ', 'ශ'], 'ර': ['රැ', 'රු'], 'ක': ['ක', 'ක්‍']
    }
    
    noisy_words = []
    actual_errors = 0
    total_chars = 0
    
    random.seed(42) # Ensure reproducibility
    
    for word in text.split():
        chars = list(word)
        has_error = False
        for i, c in enumerate(chars):
            total_chars += 1
            if c in confusions and random.random() < error_rate:
                chars[i] = random.choice(confusions[c])
                actual_errors += 1
                has_error = True
        
        noisy_word = "".join(chars)
        
        # Simulate an OCR scanner giving candidates with confidence scores
        candidates = [
            {"word": noisy_word, "confidence": random.uniform(0.7, 0.95) if has_error else 0.99},
            {"word": word, "confidence": random.uniform(0.05, 0.4) if has_error else 0.99},
            {"word": noisy_word + "ැ", "confidence": random.uniform(0.01, 0.1)}
        ]
        random.shuffle(candidates)
        noisy_words.append({"candidates": candidates, "true_word": word})
        
    return noisy_words, actual_errors, total_chars

def test_ocr_correction():
    lm = load_language_model(os.path.join(DATA_DIR, 'bigram_probabilities.json'))
    if not lm:
        print("Language model missing. Run scripts/build_bigram_model.py first.")
        return
        
    print("=== OCR Post-Correction Evaluation (Viterbi Decoder) ===")
    
    test_texts = [
        "වාත රෝග සඳහා නියඟලා අලයක් ගෙන හොඳින් සුද්ද කරගන්න",
        "පිත්ත දෝෂය කෝප වූ කල්හි ශුද්ධ ඉඟුරු ඇට පිරිසිදු කර ගල්වනු මැනවි",
        "අතීසාරය සුවපත් කරනු පිණිස අලුත් කළුදුරු චූර්ණය පොඟවා ආලේප කරනු"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1} Original: {text}")
        noisy_data, errs, chars = simulate_ocr_noise(text, error_rate=0.25) # high error rate
        
        # What the OCR would output if we just took the top confidence
        noisy_text = " ".join([max(w["candidates"], key=lambda c: c["confidence"])["word"] for w in noisy_data])
        print(f"Noisy OCR Output: {noisy_text} (Simulated CER: {errs/chars:.1%})")
        
        decoded = viterbi_decode(noisy_data, lm)
        print(f"Viterbi Decoded : {decoded}")
        
        recovered_words = sum(1 for a, b in zip(text.split(), decoded.split()) if a == b)
        noisy_correct_words = sum(1 for a, b in zip(text.split(), noisy_text.split()) if a == b)
        
        print(f"Word Recovery   : {recovered_words}/{len(text.split())} words correct (Without Viterbi: {noisy_correct_words}/{len(text.split())})")
        
if __name__ == "__main__":
    test_ocr_correction()
