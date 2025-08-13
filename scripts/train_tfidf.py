
"""
    This file train and evaluate TF‑IDF + Logistic Regression.
    Usage: python -m scripts.train_tfidf
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import EXCERPT_FILE, TRAIN_DATA, TFIDF_OUTPUT

from models.tfidf import preprocess_batch, fit_tfidf_logreg, predict, encode, save_artifacts

# model/vectorizer knobs 
SEED = 42
TFIDF_KW = dict(max_features=5000, ngram_range=(1, 2), min_df=1, max_df=1.0)
LOGREG_C = 1.0

# load the freezing json file
def _load_frozen_test_indices(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(int(i) for i in data["test_index"])

# recursively convert Path objects to strings for JSON serialization.
def serialize_meta(meta):
    if isinstance(meta, dict):
        return {k: serialize_meta(v) for k, v in meta.items()}
    elif isinstance(meta, list):
        return [serialize_meta(v) for v in meta]
    elif isinstance(meta, Path):
        return str(meta)
    else:
        return meta


def main():
    os.makedirs(TFIDF_OUTPUT, exist_ok=True)

    # load data
    df = pd.read_csv(EXCERPT_FILE).reset_index(drop=True)
    assert {"excerpt", "author"}.issubset(df.columns), "CSV must contain 'excerpt' and 'author'."

    # build train/test using frozen split 
    test_idx = _load_frozen_test_indices(TRAIN_DATA)
    mask = np.zeros(len(df), dtype=bool)
    mask[test_idx] = True
    train_df = df.loc[~mask].reset_index(drop=True)
    test_df = df.loc[mask].reset_index(drop=True)
    split_mode = f"frozen({TRAIN_DATA})"

    # preprocess
    X_train = preprocess_batch(train_df["excerpt"].tolist())
    X_test = preprocess_batch(test_df["excerpt"].tolist())
    y_train = train_df["author"].tolist()
    y_test = test_df["author"].tolist()

    # train
    vec, clf = fit_tfidf_logreg(X_train, y_train, seed=SEED, tfidf_kwargs=TFIDF_KW, C=LOGREG_C)

    # evaluate
    preds = predict(vec, clf, X_test)
    acc = accuracy_score(y_test, preds)
    mf1 = f1_score(y_test, preds, average="macro")
    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print("\n--- TF‑IDF + Logistic Regression ---")
    print(f"Split     : {split_mode}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Macro‑F1  : {mf1:.4f}")
    print(report)
    print("Confusion matrix:\n", cm)

    # 6) Save artifacts and outputs
    save_artifacts(TFIDF_OUTPUT, vec, clf)

    pd.DataFrame({"text": test_df["excerpt"], "author": y_test, "pred": preds}).to_csv(
        os.path.join(TFIDF_OUTPUT, "preds.csv"), index=False
    )
    with open(os.path.join(TFIDF_OUTPUT, "classification_report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(TFIDF_OUTPUT, "metrics.json"), "w") as f:
        json.dump({"accuracy": float(acc), "macro_f1": float(mf1), "split": split_mode}, f, indent=2)

    # dense embeddings for t‑SNE
    try:
        Z = encode(vec, X_test, svd_dim=128, seed=SEED)
        np.save(os.path.join(TFIDF_OUTPUT, "emb_test.npy"), Z.astype(np.float32))
    except Exception as e:
        print(f"(Skipping emb_test.npy: {e})")

    # save meta for reproducibility
    meta = {
        "data": EXCERPT_FILE,
        "split_file": TRAIN_DATA if os.path.exists(TRAIN_DATA) else None,
        "seed": SEED,
        "tfidf": TFIDF_KW,
        "logreg": {"C": LOGREG_C},
        "n_test": len(y_test),
    }
    with open("results/tfidf/meta.json", "w", encoding="utf-8") as f:
        json.dump(serialize_meta(meta), f, indent=2)

    print(f"\nSaved artifacts and metrics to: {TFIDF_OUTPUT}")

if __name__ == "__main__":
    main()
