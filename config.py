"""
Canonical configuration for the Ayurvedic NLP Pipeline.
All suffix lists, hyperparameters, and shared constants are defined here
to eliminate mismatch bugs across training, inference, and weak supervision.
"""
import unicodedata

# ==============================================================================
# CANONICAL MORPHOLOGICAL SUFFIX LIST
# These sentence-ending suffixes are characteristic of Sinhala SOV verb-final
# grammar in Ayurvedic instructional texts. This is the SINGLE SOURCE OF TRUTH
# used across training, inference, and weak supervision.
#
# Linguistic basis:
#   - Finite verb endings (declarative): යි, වේ, යේය, කරයි
#   - Imperative/prescriptive (clinical): මැනවි, ගනු, තබන්න, ගන්න, න්න
#   - Obligation markers: යුතු
#   - Temporal postpositions (clause-final): පෙර, පසු
#   - General suffix patterns: ස්
# ==============================================================================

CANONICAL_ENDINGS = [
    'යි',       # -yi: present tense declarative ("...කරයි" = does)
    'ස්',       # -s: various verb finals
    'යුතු',     # yuthu: obligation ("must/should")
    'යේය',      # -yēya: classical past tense
    'වේ',       # -wē: copula/become
    'මැනවි',    # manawi: formal imperative ("it is good to")
    'ගනු',      # ganu: imperative ("take")
    'පෙර',      # pera: temporal postposition ("before")
    'පසු',      # pasu: temporal postposition ("after")
    'කරයි',     # karayi: "does" (present tense finite)
    'න්න',      # -nna: colloquial imperative ending
    'ගන්න',     # ganna: "take" (colloquial imperative)
    'තබන්න',   # thabanna: "keep" (colloquial imperative)
]

# Extended endings used ONLY in weak supervision label generation.
# These are NOT used as CRF features to avoid circularity.
WEAK_SUPERVISION_EXTRA_ENDINGS = [
    'වස්',      # was: "for the purpose of"
    'නෑ',       # nǣ: "not" (negative, colloquial)
    'නැත',      # natha: "not" (formal negative)
]

# Full weak supervision set = canonical + extra
WEAK_SUPERVISION_ENDINGS = CANONICAL_ENDINGS + WEAK_SUPERVISION_EXTRA_ENDINGS


# ==============================================================================
# CRF HYPERPARAMETERS
# ==============================================================================
CRF_PARAMS = {
    'algorithm': 'lbfgs',
    'c1': 0.1,           # L1 regularisation (sparsity)
    'c2': 0.1,           # L2 regularisation (weight explosion prevention)
    'max_iterations': 100,
    'all_possible_transitions': True,
}

CRF_CHUNK_SIZE = 30          # Words per training sequence
CRF_TRAIN_SPLIT = 0.8       # 80% train / 20% test
CRF_THRESHOLD_DEFAULT = 0.15


# ==============================================================================
# VITERBI / OCR HYPERPARAMETERS
# ==============================================================================
VITERBI_ALPHA = 0.6          # OCR confidence weight
VITERBI_BETA = 0.4           # Language model weight
VITERBI_SMOOTHING = 1e-6     # Additive smoothing for unseen bigrams (log-space safe)


# ==============================================================================
# ROBERTA HYPERPARAMETERS
# ==============================================================================
ROBERTA_MODEL_NAME = 'xlm-roberta-base'
ROBERTA_LEARNING_RATE = 2e-5
ROBERTA_TRAIN_BATCH = 16
ROBERTA_EVAL_BATCH = 32
ROBERTA_EPOCHS = 3
ROBERTA_WEIGHT_DECAY = 0.01
ROBERTA_MAX_LENGTH = 128


# ==============================================================================
# SAFETY GUARDRAIL PARAMETERS
# ==============================================================================
SAFETY_DEFAULT_WINDOW = 1
SAFETY_MAX_WINDOW = 3


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def normalize_sinhala(text: str) -> str:
    """Apply NFC Unicode normalization for consistent Sinhala text handling.
    
    Sinhala Unicode (U+0D80–U+0DFF) can have multiple representations for the
    same visual character. NFC normalization ensures consistent matching.
    """
    return unicodedata.normalize('NFC', text)
