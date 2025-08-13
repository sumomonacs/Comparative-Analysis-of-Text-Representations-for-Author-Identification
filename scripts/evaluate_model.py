"""
    This file evaluates models on unseen (out-of-domain) set:
    - Compute Top-1 / Top-2 accuracy
    - Make t-SNE plots
"""

import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import (
    INFERENCE_FILE,
    TFIDF_OUTPUT, STYLO_OUTPUT, W2V_OUTPUT, LSTM_OUTPUT, SBERT_OUTPUT,
)

# model registry
MODELS = {
    "tfidf": {
        "type": "sklearn_prob",
        "artifacts": TFIDF_OUTPUT,
        "clf_keys": ["classifier.joblib", "logreg.joblib", "clf.joblib"],
        "vec_keys": ["vectorizer.joblib", "tfidf_vectorizer.joblib"],
        "scaler_keys": [],
    },
    "stylometry": {
        "type": "sklearn_prob",
        "artifacts": STYLO_OUTPUT,
        "clf_keys": ["logreg.joblib", "classifier.joblib", "clf.joblib"],
        "vec_keys": ["stylometry_vectorizer.joblib", "vectorizer.joblib"],
        "scaler_keys": ["scaler.joblib"],
    },
    "word2vec": {
        "type": "sklearn_prob",
        "artifacts": W2V_OUTPUT,
        "clf_keys": ["classifier.joblib", "mlp.joblib", "clf.joblib", "logreg.joblib"],
        "vec_keys": ["vectorizer.joblib", "w2v_vectorizer.joblib", "idf_vectorizer.joblib"],
        "scaler_keys": ["scaler.joblib"],
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
    },
}
OUT_ROOT = Path("results/eval_unseen")
SEED = 42

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
    assert not miss, f"inference CSV is missing: {miss}"
    return df.reset_index(drop=True)

def _topk_hits(y_true, y_topk, k):
    hits = sum(1 for gt, cand in zip(y_true, y_topk) if gt in cand[:k])
    return hits / max(1, len(y_true))

def _safe_tsne(Z, labels, title, out_png):
    Z = np.asarray(Z)
    if Z.ndim != 2:
        Z = Z.reshape(len(Z), -1)
    if Z.dtype.kind not in ("f", "c"):
        Z = Z.astype(np.float32, copy=False)
    if Z.shape[1] > 50:
        Z = TruncatedSVD(n_components=50, random_state=SEED).fit_transform(Z).astype(np.float32, copy=False)

    perplexity = max(5, min(30, (len(Z) - 1) // 3)) if len(Z) > 10 else max(2, (len(Z) - 1) // 3 or 2)
    tsne = TSNE(n_components=2, random_state=SEED, init="pca", learning_rate=200.0, perplexity=perplexity)
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

def _eval_sklearn_prob(spec, excerpts):
    import joblib
    from scipy import sparse as _sp
    art = spec["artifacts"]

    clf_path = _find_first(art, spec.get("clf_keys", []))
    assert clf_path, f"No classifier joblib found in {art}"
    clf = joblib.load(clf_path)

    vec = scaler = None
    if hasattr(clf, "predict_proba") and not hasattr(clf, "steps"):
        vp = _find_first(art, spec.get("vec_keys", []))
        spath = _find_first(art, spec.get("scaler_keys", []))
        if vp: vec = joblib.load(vp)
        if spath: scaler = joblib.load(spath)

    if vec is not None:
        X = vec.transform(excerpts)
        if scaler is not None:
            X = scaler.transform(X.toarray() if _sp.issparse(X) else X)
    else:
        X = excerpts

    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X)
        classes = clf.classes_.tolist()
    else:
        scores = clf.decision_function(X)
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        prob = e / e.sum(axis=1, keepdims=True)
        classes = getattr(clf, "classes_", list(range(prob.shape[1])))

    # embed from prob via SVD
    k = 2 if prob.shape[1] <= 2 else min(50, prob.shape[1] - 1)
    embed = TruncatedSVD(n_components=k, random_state=SEED).fit_transform(prob).astype(np.float32, copy=False)
    return classes, prob, embed

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

    embed = encode_texts(model, excerpts, stoi, classes, cfg).astype(np.float32, copy=False)
    return classes, prob, embed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="tfidf,stylometry,word2vec,lstm,sbert")
    args = ap.parse_args()

    df = _load_unseen()
    excerpts = df["excerpt"].tolist()
    gold = df["author"].tolist()

    for name in [m.strip() for m in args.models.split(",")]:
        if name not in MODELS:
            print(f"[skip] unknown model: {name}")
            continue
        spec = MODELS[name]
        if not os.path.exists(spec["artifacts"]):
            print(f"[skip] {name}: artifacts not found at {spec['artifacts']}")
            continue

        print(f"\n=== Evaluating {name} on unseen set ===")
        if spec["type"] == "lstm":
            classes, prob, embed = _eval_lstm(spec, excerpts)
        else:
            classes, prob, embed = _eval_sklearn_prob(spec, excerpts)

        order = np.argsort(-prob, axis=1)
        topk = [[classes[j] for j in row[:3]] for row in order]
        top1_acc = float(np.mean([g == row[0] for g, row in zip(gold, topk)]))
        top2_acc = float(_topk_hits(gold, topk, k=2))

        print(f" Top-1 acc: {top1_acc:.4f}")
        print(f" Top-2 acc: {top2_acc:.4f}")

        out_png = OUT_ROOT / name / "tsne_unseen.png"
        _safe_tsne(embed, gold, title=f"{name} — t-SNE (unseen)", out_png=out_png)
        print(f" Saved t-SNE to {out_png}")

if __name__ == "__main__":
    main()
