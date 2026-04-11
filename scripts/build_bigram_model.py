import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import json
from collections import defaultdict, Counter
from config import normalize_sinhala, DATA_DIR

def build_bigram_model(file_path):
    print(f"Reading {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Make sure the file name is correct.")
        return None

    # Dictionaries to hold our counts
    bigram_counts = defaultdict(Counter)
    unigram_counts = Counter()

    print("Counting words and pairs...")
    for line in lines:
        # Split the sentence into individual words
        words = line.strip().split()
        if not words:
            continue

        # Loop through the sentence to count pairs
        for i in range(len(words) - 1):
            word1 = normalize_sinhala(words[i])
            word2 = normalize_sinhala(words[i+1])
            
            unigram_counts[word1] += 1
            bigram_counts[word1][word2] += 1
        
        # Add the very last word of the sentence to the unigram count
        if words:
            unigram_counts[words[-1]] += 1

    print("Calculating probabilities...")
    probabilities = {}
    
    for word1, next_words in bigram_counts.items():
        probabilities[word1] = {}
        # How many times did word1 appear in total?
        total_word1 = unigram_counts[word1] 
        
        for word2, count in next_words.items():
            # Calculate the percentage/probability
            prob = count / total_word1
            probabilities[word1][word2] = round(prob, 5) # Round to 5 decimal places

    return probabilities

# --- Execution ---
input_corpus = os.path.join(DATA_DIR, 'cleaned_corpus.txt')
output_json = os.path.join(DATA_DIR, 'bigram_probabilities.json')

print("Starting Language Model Training...")
model = build_bigram_model(input_corpus)

if model:
    # Save the model to a JSON file so we don't have to retrain it every time
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    print(f"\nSuccess! Language Model saved to '{output_json}'")
    print(f"Total unique starting words learned: {len(model)}")