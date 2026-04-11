import csv
import os
import sys

# Allow imports from src directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from config import WEAK_SUPERVISION_ENDINGS, normalize_sinhala

# ==============================================================================
# WEAK SUPERVISION SCRIPT
# Purpose: Automatically generate a labeled training dataset (train_labeled.tsv)
#          from an unpunctuated classical Sinhala text corpus.
# Method:  Uses linguistic morphological rules (verb endings) to programmatically
#          inject 'STOP' (boundary) or 'O' (continue) tags without human labeling.
# ==============================================================================

# Input and Output Paths
INPUT_CORPUS = os.path.join(os.path.dirname(__file__), "..", "train.txt")
OUTPUT_TSV = "train_labeled.tsv"

# Endings imported from config.py (canonical source of truth)
ENDINGS = WEAK_SUPERVISION_ENDINGS

def generate_labeled_data(input_file, output_file):
    print(f"🔄 Starting Weak Supervision Generation...")
    print(f"📖 Reading raw corpus from '{input_file}'")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found.")
        return

    labeled_data = []
    sentence_count = 0
    total_words = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        # Read lines, each line in train.txt is assumed to be a complete thought/sentence 
        # (or at least, the end of the line is a definite sentence boundary).
        # We will tag the words. If a word matches an ending rule, we tag it 'STOP'.
        # Note: If the raw text doesn't have newlines per sentence, this script will 
        # still apply the morphological rules to find stops.
        
        for line in f:
            words = line.strip().split()
            if not words:
                continue
            
            sentence_count += 1
            for i, word in enumerate(words):
                total_words += 1
                # Check if the word ends with any of our morphological features
                is_boundary = False
                for ending in ENDINGS:
                    if word.endswith(ending):
                        is_boundary = True
                        break
                
                # Alternatively, if it's the absolute last word on the line in our corpus, 
                # we force it to be a STOP to ensure we capture the implicit line breaks.
                if i == len(words) - 1:
                    is_boundary = True
                
                tag = "STOP" if is_boundary else "O"
                labeled_data.append((word, tag))
            
            # Add a blank line to separate sequences (CRF expects this)
            labeled_data.append(("", ""))

    print(f"📝 Generated tags for {total_words} words across {sentence_count} sequences.")
    print(f"💾 Saving to '{output_file}'...")

    # Write to TSV format compatible with CoNLL/CRFSuite
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for word, tag in labeled_data:
            if word == "" and tag == "":
                f.write("\n") # Sequence separator
            else:
                writer.writerow([word, tag])
                
    print(f"✅ Success! Labeled dataset generated at '{output_file}'.")
    print(f"📊 Ready for Phase 2: CRF or Transformer Training.")

if __name__ == "__main__":
    generate_labeled_data(INPUT_CORPUS, OUTPUT_TSV)
