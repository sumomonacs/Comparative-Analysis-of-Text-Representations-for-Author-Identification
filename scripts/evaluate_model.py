
"""
    This file defines a common evaluator for unseen classification (Top‑1 / Top‑3) + t‑SNE plots.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import (
    INFERENCE_FILE,
    TFIDF_OUTPUT, STYLO_OUTPUT, W2V_OUTPUT, LSTM_OUTPUT, SBERT_OUTPUT,
)


# model register 
MODELS = {
    "tfidf": {
        "type": "sklearn_prob",
        "artifacts": TFIDF_OUTPUT,
        "clf_keys": ["classifier.joblib", "logreg.joblib", "clf.joblib"],
        "vec_keys": ["vectorizer.joblib", "tfidf_vectorizer.joblib"],
        "scaler_keys": [],
        "featurizer_keys": [],
    },
    "stylometry": {
        "type": "sklearn_prob",
        "artifacts": STYLO_OUTPUT,
        "clf_keys": ["logreg.joblib", "classifier.joblib", "clf.joblib"],
        "vec_keys": ["stylometry_vectorizer.joblib", "vectorizer.joblib"],
        "scaler_keys": ["scaler.joblib"],
        "featurizer_keys": [],
    },
    "word2vec": {
        "type": "sklearn_prob",
        "artifacts": W2V_OUTPUT,
        "clf_keys": ["classifier.joblib", "mlp.joblib", "clf.joblib", "logreg.joblib"],
        "vec_keys": ["vectorizer.joblib", "w2v_vectorizer.joblib", "idf_vectorizer.joblib"],
        "scaler_keys": ["scaler.joblib"],
        "featurizer_keys": [],
    },
    "lstm": {
        "type": "lstm",
        "artifacts": LSTM_OUTPUT,
    },
    "sbert": {
        "type": "sklearn_prob",
        "artifacts": SBERT_OUTPUT,
        "clf_keys": ["classifier.joblib", "logreg.joblib", "clf.joblib"],
        "vec_keys": ["vectorizer.joblib", "sbert_vectorizer.joblib"],
        "scaler_keys": ["scaler.joblib"],
        "featurizer_keys": [],
    },
}

OUT_ROOT = Path("results/eval")
SEED = 42

# helpers
def _find_first(art_dir, candidates):
    for name in candidates:
        p = os.path.join(art_dir, name)
        if os.path.exists(p):
            return p
    return None

def _load_unseen():
    df = pd.read_csv(INFERENCE_FILE)
    required = {"excerpt", "author", "book", "bucket"}
    miss = required - set(df.columns)
    assert not miss, "inference CSV is missing: %s" % miss
    return df.reset_index(drop=True)

def _topk_hits(y_true, y_topk, k):
    hits = sum(1 for gt, cand in zip(y_true, y_topk) if gt in cand[:k])
    return hits / max(1, len(y_true))


# adapters
def _eval_sklearn_prob(spec, excerpts):
    import joblib
    art = spec["artifacts"]

    clf_path = _find_first(art, spec.get("clf_keys", ["classifier.joblib", "clf.joblib", "logreg.joblib"]))
    assert clf_path, "No classifier joblib found in %s" % art
    clf = joblib.load(clf_path)

    vec = None
    if "vec_keys" in spec:
        vec_path = _find_first(art, spec["vec_keys"])
        if vec_path:
            vec = joblib.load(vec_path)

    scaler = None
    if "scaler_keys" in spec:
        scaler_path = _find_first(art, spec["scaler_keys"])
        if scaler_path:
            scaler = joblib.load(scaler_path)

    if vec is not None:
        X = vec.transform(excerpts)
        if scaler is not None:
            try:
                X = scaler.transform(X)
            except Exception:
                X = scaler.transform(X.toarray())
    else:
        X = excerpts  # hope the clf is a Pipeline or can handle raw text

    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X)
        classes = clf.classes_.tolist()
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        prob = e / e.sum(axis=1, keepdims=True)
        classes = getattr(clf, "classes_", list(range(prob.shape[1])))
    else:
        raise RuntimeError("Classifier exposes neither predict_proba nor decision_function")

    emb_npy = os.path.join(art, "emb_test.npy")
    if os.path.exists(emb_npy):
        embed = np.load(emb_npy).astype(np.float32, copy=False)  # <-- add astype
    else:
        if hasattr(X, "shape"):
            k = 2
            if X.shape[1] > 1:
                k = min(50, X.shape[1] - 1)
            embed = TruncatedSVD(n_components=max(2, k), random_state=SEED).fit_transform(X).astype(np.float32, copy=False)  
        else:
            k = 2
            if prob.shape[1] > 1:
                k = min(50, prob.shape[1] - 1)
            embed = TruncatedSVD(n_components=max(2, k), random_state=SEED).fit_transform(prob).astype(np.float32, copy=False)  

    order = np.argsort(-prob, axis=1)
    topk = [[classes[j] for j in row[:3]] for row in order]



    return classes, topk, prob, embed

def _eval_lstm(spec, excerpts):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from models.lstm import load_artifacts, TextDataset, pad_collate, encode_texts

    art = spec["artifacts"]
    model, stoi, classes, cfg = load_artifacts(art)
    device = cfg.device if isinstance(cfg.device, str) else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ds = TextDataset(excerpts, [classes[0]] * len(excerpts), stoi, classes,
                     lowercase=cfg.lowercase, alpha_only=cfg.alpha_only, max_len=cfg.max_len)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)

    probs = []
    with torch.no_grad():
        for xb, _, lengths in dl:
            xb, lengths = xb.to(device), lengths.to(device)
            logits, _ = model(xb, lengths)
            p = F.softmax(logits, dim=-1).cpu().numpy()
            probs.append(p)
    prob = np.vstack(probs)
    order = np.argsort(-prob, axis=1)
    topk = [[classes[j] for j in row[:3]] for row in order]

    emb_npy = os.path.join(art, "emb_test.npy")
    if os.path.exists(emb_npy):
        embed = np.load(emb_npy)
    else:
        embed = encode_texts(model, excerpts, stoi, classes, cfg)

    return classes, topk, prob, embed


# plot
def _plot_tsne(Z, labels, title, out_png):
    import numpy as np
    from sklearn.manifold import TSNE
    from sklearn.decomposition import TruncatedSVD
    import matplotlib.pyplot as plt

    Z = np.asarray(Z)
    # coerce to float 
    if Z.dtype.kind not in ("f", "c"):
        Z = Z.astype(np.float32, copy=False)

    # light SVD pre-reduction for very high dims
    if Z.ndim != 2:
        Z = Z.reshape(len(Z), -1)
    if Z.shape[1] > 50:
        Z = TruncatedSVD(n_components=50, random_state=SEED).fit_transform(Z)

    tsne = TSNE(n_components=2, random_state=SEED, init="pca", learning_rate="auto")
    Y = tsne.fit_transform(Z)

    authors = sorted(set(labels))
    color_map = {a: i for i, a in enumerate(authors)}
    colors = [color_map[a] for a in labels]

    plt.figure(figsize=(7, 6))
    plt.scatter(Y[:, 0], Y[:, 1], c=colors, s=10, alpha=0.85)
    handles = [plt.Line2D([0], [0], marker='o', linestyle='', markersize=6, label=a) for a in authors]
    plt.legend(handles, authors, loc="best", fontsize=8, frameon=True)
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()



#one model
def _evaluate_one(name, spec, df):
    art = spec["artifacts"]
    if not os.path.exists(art):
        print(f"[skip] {name}: artifacts not found at {art}")
        return

    print(f"\n=== Evaluating {name} ===")
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    excerpts = df["excerpt"].tolist()
    gold = df["author"].tolist()
    buckets = df["bucket"].tolist()

    if spec["type"] == "sklearn_prob":
        classes, topk, prob, embed = _eval_sklearn_prob(spec, excerpts)
    elif spec["type"] == "lstm":
        classes, topk, prob, embed = _eval_lstm(spec, excerpts)
    else:
        raise ValueError("Unknown model type: %s" % spec["type"])

    top1 = [row[0] for row in topk]
    top1_acc = float(np.mean([g == p for g, p in zip(gold, top1)]))
    top3_acc = float(_topk_hits(gold, topk, k=3))

    df_pred = pd.DataFrame({
        "excerpt": excerpts,
        "author": gold,
        "bucket": buckets,
        "pred_top1": top1,
        "in_top3": [g in row for g, row in zip(gold, topk)],
    })
    by_bucket = (
        df_pred.groupby("bucket")
        .agg(
            n=("author", "size"),
            top1_acc=("pred_top1", lambda s: float(np.mean(df_pred.loc[s.index, "author"] == s))),
            top3_acc=("in_top3", "mean"),
        )
        .reset_index()
    )

    df_pred.to_csv(out_dir / "preds.csv", index=False)
    by_bucket.to_csv(out_dir / "metrics_by_bucket.csv", index=False)
    with open(out_dir / "metrics_overall.json", "w") as f:
        json.dump({"model": name, "n": len(gold), "top1_acc": top1_acc, "top3_acc": top3_acc}, f, indent=2)

    print(f" Overall: top1={top1_acc:.4f}  top3={top3_acc:.4f}")
    print(" By bucket:")
    print(by_bucket.to_string(index=False))

    _plot_tsne(embed, gold, title=f"{name} — t‑SNE (unseen)", out_png=out_dir / "tsne.png")
    print(f" Saved: {out_dir/'preds.csv'}, {out_dir/'metrics_by_bucket.csv'}, "
          f"{out_dir/'metrics_overall.json'}, {out_dir/'tsne.png'}")


def main():
    df = _load_unseen()
    print("Authors in unseen set:", sorted(df["author"].unique().tolist()))
    for name, spec in MODELS.items():
        _evaluate_one(name, spec, df)

if __name__ == "__main__":
    main()
