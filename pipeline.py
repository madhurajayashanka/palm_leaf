import joblib
import csv

# ==========================================
# PHASE 2: SEGMENTER (වාක්‍ය ඛණ්ඩනය)
# ==========================================
def word2features(sent, i):
    word = sent[i][0]
    
    # නව ක්‍රියාපද අවසානයන් (Suffixes) මෙහි එකතු කර ඇත
    common_endings = ['යි', 'ස්', 'යුතු', 'යේය', 'වේ', 'මැනවි', 'ගනු', 'පෙර', 'පසු', 'කරයි', 'න්න', 'ගන්න', 'තබන්න']
    
    is_ending = any(word.endswith(suffix) for suffix in common_endings)
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-2:]': word[-2:], 
        'word[-3:]': word[-3:], 
        'word.length': len(word),
        'is_common_ending': is_ending, 
    }
    if i > 0:
        features.update({'-1:word.lower()': sent[i-1][0].lower(), '-1:word[-2:]': sent[i-1][0][-2:]})
    else:
        features['BOS'] = True 
    if i < len(sent) - 1:
        features.update({'+1:word.lower()': sent[i+1][0].lower()})
    else:
        features['EOS'] = True 
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def segment_text(text, model_path="ayurvedic_segmenter.pkl", threshold=0.15):
    try:
        crf = joblib.load(model_path)
    except FileNotFoundError:
        return "Error: Segmenter Model not found."

    words = text.strip().split()
    if not words: return ""

    dummy_sent = [(w, "") for w in words]
    features = [sent2features(dummy_sent)]
    marginals = crf.predict_marginals(features)[0]

    segmented_words = []
    for i, word in enumerate(words):
        prob_stop = marginals[i].get('STOP', 0)
        if prob_stop > threshold:
            segmented_words.append(word + ".")
        else:
            segmented_words.append(word)

    return " ".join(segmented_words)


# ==========================================
# PHASE 3: KNOWLEDGE GRAPH (ආරක්ෂක පද්ධතිය)
# ==========================================
def load_knowledge_graph(csv_filepath="ayurvedic_ingredients_full.csv"):
    kg = {}
    try:
        with open(csv_filepath, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                entity = row["Entity"].strip()
                toxicity = row["Toxicity"].strip().lower()
                
                # විෂ නැති ඖෂධ මඟ හැරීම
                if toxicity in ['low', 'none', 'safe', 'no', '']: continue 
                
                aliases = [a.strip() for a in row["Aliases"].split(",") if a.strip()]
                purification = [p.strip() for p in row["Purification_Keywords"].split(",") if p.strip()]
                
                if not purification: continue
                
                kg[entity] = {
                    "aliases": aliases,
                    "toxicity": row["Toxicity"].strip(),
                    "shodhana_keywords": purification
                }
        return kg
    except FileNotFoundError:
        return None

def analyze_safety(segmented_text, kg, window_size=1):
    sentences = [s.strip() for s in segmented_text.split('.') if s.strip()]
    issues_found = 0
    report_details = [] # UI එකට යැවීමට වාර්තාවක් සෑදීම

    all_toxic_terms = []
    term_to_data = {}
    
    for entity, data in kg.items():
        for t in [entity] + data["aliases"]:
            if t not in term_to_data:
                all_toxic_terms.append(t)
                term_to_data[t] = (entity, data)
                
    all_toxic_terms.sort(key=len, reverse=True)

    for i, sentence in enumerate(sentences):
        temp_sentence = f" {sentence} "
        found_items = []
        
        for term in all_toxic_terms:
            if f" {term} " in temp_sentence:
                main_entity, data = term_to_data[term]
                found_items.append((term, main_entity, data))
                temp_sentence = temp_sentence.replace(f" {term} ", " [MASK] ") # Masking Bug Fix
                
        for term, main_entity, data in found_items:
            start_index = max(0, i - window_size)
            end_index = min(len(sentences), i + window_size + 1)
            context_block = " ".join(sentences[start_index:end_index])
            
            has_shodhana = any(keyword in context_block for keyword in data["shodhana_keywords"])
            
            if has_shodhana:
                report_details.append({
                    "term": term, "sentence_id": i+1, "status": "PASS", 
                    "message": f"✅ '{term}' හඳුනාගන්නා ලදී. ශෝධන ක්‍රමවේදය අඩංගු බැවින් ආරක්ෂිතයි."
                })
            else:
                report_details.append({
                    "term": term, "sentence_id": i+1, "status": "ALERT", 
                    "message": f"🔴 අනතුරු ඇඟවීමයි! '{term}' අඩංගු වුවද ශෝධන (පිරිසිදු කිරීමේ) උපදෙස් නොමැත."
                })
                issues_found += 1

    final_status = "APPROVED" if issues_found == 0 else "REJECTED"
    
    return {
        "final_status": final_status,
        "issues_count": issues_found,
        "details": report_details
    }