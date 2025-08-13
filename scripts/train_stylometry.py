"""
    This file trains and evaluates Stylometry + Logistic Regression.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import EXCERPT_FILE, TRAIN_DATA, STYLO_OUTPUT

from models.stylometry import StylometricVectorizer  

# knobs
SEED = 42
MAX_FUNCTION_WORDS = 300
LOGREG_C = 0.1


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

# save fitted components as Joblib files 
def save_artifacts(out_dir, vec, scaler, clf):
    from joblib import dump
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump(vec, out_dir / "stylometry_vectorizer.joblib")
    dump(scaler, out_dir / "scaler.joblib")
    dump(clf, out_dir / "logreg.joblib")


def main():
    os.makedirs(STYLO_OUTPUT, exist_ok=True)

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

    # fit vectorizer 
    vec = StylometricVectorizer(max_function_words=MAX_FUNCTION_WORDS)
    vec.fit(X_train)
    F_train = vec.transform(X_train)
    F_test  = vec.transform(X_test)

    # scale dense features 
    scaler = StandardScaler(with_mean=True, with_std=True)
    F_train_std = scaler.fit_transform(F_train)
    F_test_std  = scaler.transform(F_test)

    # train classifier
    clf = LogisticRegression(max_iter=10000, multi_class="multinomial", C=LOGREG_C, random_state=SEED)
    clf.fit(F_train_std, y_train)

    # evaluate
    preds = clf.predict(F_test_std)
    acc = accuracy_score(y_test, preds)
    mf1 = f1_score(y_test, preds, average="macro")
    report_txt = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print("\n--- Stylometry + Logistic Regression ---")
    print(f"Split     : {split_mode}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Macro‑F1  : {mf1:.4f}")
    print(report_txt)
    print("Confusion matrix:\n", cm)

    # save artifacts and outputs
    save_artifacts(STYLO_OUTPUT, vec, scaler, clf)

    # predictions table
    pd.DataFrame({
        "text": test_df["excerpt"],
        "author": y_test,
        "pred": preds
    }).to_csv(os.path.join(STYLO_OUTPUT, "preds.csv"), index=False)

    # reports
    with open(os.path.join(STYLO_OUTPUT, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)
    with open(os.path.join(STYLO_OUTPUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"accuracy": float(acc), "macro_f1": float(mf1), "split": split_mode}, f, indent=2)

    # save meta for reproducibility
    meta = {
        "data": EXCERPT_FILE,
        "split_file": TRAIN_DATA,
        "seed": SEED,
        "stylometry": {
            "max_function_words": MAX_FUNCTION_WORDS,
            "include_punct": True,
            "include_function_word_vector": True,
        },
        "scaler": "StandardScaler(with_mean=True, with_std=True)",
        "logreg": {"C": LOGREG_C, "max_iter": 10000, "multi_class": "multinomial"},
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    Path(STYLO_OUTPUT).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(STYLO_OUTPUT, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(serialize_meta(meta), f, indent=2)

    print(f"\nSaved artifacts and metrics to: {STYLO_OUTPUT}")


if __name__ == "__main__":
    main()
