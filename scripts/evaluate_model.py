import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import joblib
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import (
    INFERENCE_FILE,
    TFIDF_OUTPUT, STYLO_OUTPUT, W2V_OUTPUT, LSTM_OUTPUT, SBERT_OUTPUT,
)

# --------- registry ---------
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
        "type": "sbert_infer",        
        "artifacts": SBERT_OUTPUT,
    },

}
OUT_ROOT = Path("results/eval_unseen")
SEED = 42

# --------- utils ---------
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

    n = Z.shape[0]
    assert n == len(labels), f"length mismatch: Z={n} labels={len(labels)}"
    perplexity = max(5, min(30, (n - 1) // 3)) if n > 10 else max(2, (n - 1) // 3 or 2)
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

# --------- adapters ---------
def _eval_sklearn_prob(spec, excerpts):

    art = spec["artifacts"]

    # prefer a full pipeline that accepts raw text
    clf_path = _find_first(art, ["pipeline.joblib", "classifier.joblib", "model.joblib", "clf.joblib", "logreg.joblib"])
    if not clf_path:
        raise FileNotFoundError(f"No classifier pipeline found in {art}")
    clf = joblib.load(clf_path)

    # get probabilities directly from raw text
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(excerpts)
        classes = getattr(clf, "classes_", None)
        if classes is None and hasattr(clf, "steps"):
            classes = getattr(clf.steps[-1][1], "classes_", None)
        classes = classes.tolist() if classes is not None else list(range(prob.shape[1]))
    else:
        # rare fallback: decision_function
        scores = clf.decision_function(excerpts)
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        prob = e / e.sum(axis=1, keepdims=True)
        classes = getattr(clf, "classes_", list(range(prob.shape[1])))

    # make t‑SNE embedding from current probs (always length-matched)
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

    # Fresh embeddings for THIS batch
    embed = encode_texts(model, excerpts, stoi, classes, cfg).astype(np.float32, copy=False)

    return classes, prob, embed

def _eval_sbert_infer(spec, excerpts):
    """
    SBERT embed-only path that mirrors scripts.infer_author:
    - load best.pt/last.pt
    - build SiameseModel with same constructor args
    - model.encode(texts, batch_size=32)
    Returns (classes=None, prob=None, embed)
    """
    import torch
    from pathlib import Path
    from models.sbert import SiameseModel

    art = Path(spec["artifacts"])
    ckpt = art / "best.pt"
    if not ckpt.exists():
        alt = art / "last.pt"
        assert alt.exists(), f"No SBERT checkpoint in {art} (looked for best.pt/last.pt)"
        ckpt = alt

    # load checkpoint (PyTorch 2.6-safe)
    try:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(str(ckpt), map_location="cpu")

    raw = state.get("model_state_dict", state)
    weights = {k.replace("module.", "", 1): v for k, v in raw.items()}

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    # build model exactly like scripts/infer_author.py
    model = SiameseModel(
        encoder_name="all-MiniLM-L6-v2",
        proj_dim=256, mlp_hidden=512, dropout=0.1, init_temp=10.0, device=device
    ).to(device)
    model.load_state_dict(weights, strict=False)
    model.eval()

    # encode excerpts (mirrors _rank_authors_batch → model.encode)
    with torch.inference_mode():
        embed = model.encode(excerpts, batch_size=32).detach().cpu().float().numpy()

    classes, prob = None, None
    return classes, prob, embed



# --------- main ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="tfidf,stylometry,word2vec,lstm,sbert")
    args = ap.parse_args()

    df = _load_unseen()
    excerpts = df["excerpt"].tolist()
    gold = df["author"].tolist()

    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
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
        elif spec["type"] == "sbert":
            classes, prob, embed = _eval_sbert_infer(spec, excerpts)
        else:
            classes, prob, embed = _eval_sklearn_prob(spec, excerpts)


        order = np.argsort(-prob, axis=1)
        topk = [[classes[j] for j in row[:3]] for row in order]
        top1 = [row[0] for row in topk]
        top1_acc = float(np.mean([g == p for g, p in zip(gold, top1)]))
        top2_acc = float(_topk_hits(gold, topk, k=2))

        print(f" Top-1 acc: {top1_acc:.4f}")
        print(f" Top-2 acc: {top2_acc:.4f}")

        out_png = OUT_ROOT / f"{name}-tsne.png"
        # final guard to ensure lengths match (they should)
        if embed.shape[0] != len(gold):
            # fallback: derive from prob so it always matches
            k = 2 if prob.shape[1] <= 2 else min(50, prob.shape[1] - 1)
            embed = TruncatedSVD(n_components=k, random_state=SEED).fit_transform(prob).astype(np.float32, copy=False)
        _safe_tsne(embed, gold, title=f"{name} — t-SNE (unseen)", out_png=out_png)
        print(f" Saved t-SNE to {out_png}")

if __name__ == "__main__":
    main()