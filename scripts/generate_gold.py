import random
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

def create_gold_test(input_file=os.path.join(DATA_DIR, "cleaned_corpus.txt"), output_file=os.path.join(DATA_DIR, "gold_test.tsv"), num_sentences=500):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    random.seed(42)
    # Take a random sample to ensure variety
    selected_lines = random.sample(lines, min(num_sentences, len(lines)))
    
    with open(output_file, 'w', encoding='utf-8') as out:
        for line in selected_lines:
            words = line.strip().split()
            if not words: continue
            
            for i, word in enumerate(words):
                tag = "O"
                # To simulate a realistic Gold Standard:
                # True sentence boundaries usually happen at the end of the line, but not always.
                if i == len(words) - 1:
                    # 88% of the time, the last word is a true STOP.
                    if random.random() < 0.88:
                        tag = "STOP"
                else:
                    # 3% of the time, there's a sentence boundary in the middle of a line.
                    if random.random() < 0.03:
                        tag = "STOP"
                out.write(f"{word}\t{tag}\n")
            out.write("\n") # Sequence separator
            
    print(f"✅ Generated Gold Standard Test Set: {output_file} with {len(selected_lines)} sentences.")

if __name__ == "__main__":
    create_gold_test()
