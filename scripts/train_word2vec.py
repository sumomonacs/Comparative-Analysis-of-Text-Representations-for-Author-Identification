"""
    This file trains and evaluates Word2Vec (doc embeddings) + Logistic Regression.
    Usage: python -m scripts.train_word2vec
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import EXCERPT_FILE, TRAIN_DATA, W2V_OUTPUT

from models.word2vec import Word2VecDocEmbed, W2VConfig


# knobs
SEED = 42
CFG = W2VConfig(
    vector_size=300,
    window=5,
    min_count=3,
    sg=1,                 # skip-gram
    negative=10,
    epochs=10,
    workers=4,
    lowercase=True,
    alpha_only=True,
    doc_pool="sif",       # "mean" | "tfidf" | "sif"
    sif_a=1e-3,
    remove_first_pc=True,
    clf_C=1.0,
    seed=SEED,
)


# --------------------------- helpers ---------------------------
# load frozen split
def _load_frozen_test_indices(json_path: str | Path) -> list[int]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(int(i) for i in data["test_index"])

# recursively convert Path objects (and numpy scalars) to JSON-serializable values
def serialize_meta(meta):
    if isinstance(meta, dict):
        return {k: serialize_meta(v) for k, v in meta.items()}
    if isinstance(meta, list):
        return [serialize_meta(v) for v in meta]
    if isinstance(meta, (np.integer, np.int32, np.int64)):
        return int(meta)
    if isinstance(meta, (np.floating, np.float32, np.float64)):
        return float(meta)
    if isinstance(meta, Path):
        return str(meta)
    return meta


def main():
    os.makedirs(W2V_OUTPUT, exist_ok=True)

    # load data
    df = pd.read_csv(EXCERPT_FILE).reset_index(drop=True)
    assert {"excerpt", "author"}.issubset(df.columns), "CSV must contain 'excerpt' and 'author'."

    # build train/test using frozen split
    test_idx = _load_frozen_test_indices(TRAIN_DATA)
    mask = np.zeros(len(df), dtype=bool)
    mask[test_idx] = True
    train_df = df.loc[~mask].reset_index(drop=True)
    test_df  = df.loc[ mask].reset_index(drop=True)
    split_mode = f"frozen({TRAIN_DATA})"

    # prepare X/y
    X_train = train_df["excerpt"].tolist()
    X_test  = test_df["excerpt"].tolist()
    y_train = train_df["author"].tolist()
    y_test  = test_df["author"].tolist()

    # train model
    model = Word2VecDocEmbed(CFG).fit(X_train, y_train)

    # evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mf1 = f1_score(y_test, preds, average="macro")
    report_txt = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print("\n--- Word2Vec (doc) + Logistic Regression ---")
    print(f"Split     : {split_mode}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Macro‑F1  : {mf1:.4f}")
    print(report_txt)
    print("Confusion matrix:\n", cm)

    # save artifacts + outputs
    artifacts_dir = os.path.join(W2V_OUTPUT, "artifacts")
    model.save(artifacts_dir)

    pd.DataFrame({
        "text": test_df["excerpt"],
        "author": y_test,
        "pred": preds
    }).to_csv(os.path.join(W2V_OUTPUT, "preds.csv"), index=False)

    with open(os.path.join(W2V_OUTPUT, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    with open(os.path.join(W2V_OUTPUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"accuracy": float(acc), "macro_f1": float(mf1), "split": split_mode}, f, indent=2)

    # save doc embeddings of test set for downstream plots
    X_test_emb = model.encode(X_test)
    np.save(os.path.join(W2V_OUTPUT, "emb_test.npy"), X_test_emb.astype(np.float32))

    # save meta for reproducibility
    meta = {
        "data": EXCERPT_FILE,
        "split_file": TRAIN_DATA,
        "seed": SEED,
        "w2v_config": {
            "vector_size": CFG.vector_size,
            "window": CFG.window,
            "min_count": CFG.min_count,
            "sg": CFG.sg,
            "negative": CFG.negative,
            "epochs": CFG.epochs,
            "doc_pool": CFG.doc_pool,
            "sif_a": CFG.sif_a,
            "remove_first_pc": CFG.remove_first_pc,
            "clf_C": CFG.clf_C,
        },
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    with open(os.path.join(W2V_OUTPUT, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(serialize_meta(meta), f, indent=2)

    print(f"\nSaved artifacts and metrics to: {W2V_OUTPUT}")

if __name__ == "__main__":
    main()
