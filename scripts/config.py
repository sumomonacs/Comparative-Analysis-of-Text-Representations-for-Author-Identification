from pathlib import Path

# DATA FILE
DATA_DIR = Path("data")

RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
EVALUATE_RAW_DIR = DATA_DIR / "evaluate_raw"
EVALUATE_CLEANED_DIR = DATA_DIR / "evaluate_cleaned"

EXCERPT_FILE = DATA_DIR / "excerpts.csv"
INFERENCE_FILE = DATA_DIR / "inference.csv"
PAIR_FILE = DATA_DIR / "pairs.json"
TRAIN_DATA = DATA_DIR / "train.json"

# RESULT FILE
TFIDF_OUTPUT = "results/tfidf"
STYLO_OUTPUT = "results/stylometry"
W2V_OUTPUT = "results/word2vec"
LSTM_OUTPUT = "results/lstm"
SBERT_OUTPUT = "results/sbert"

# EVALUATION FILE
EVOLVE_DIR = "results/eval_metrics"
PIC_DIR = "results/eval_unseen"