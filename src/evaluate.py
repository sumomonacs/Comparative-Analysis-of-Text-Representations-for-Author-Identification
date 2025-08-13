from pathlib import Path
import json                         
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from src.model import PairDataset, SiameseModel

# loader 
def _make_loader(pairs_json, batch_size):
    ds = PairDataset(pairs_json)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)

# embed helpers
def _embed_texts(model, texts, device, batch=128):
    embs = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            e = model._embed(texts[i:i+batch]).to("cpu").numpy()
            embs.append(e)
    X = np.vstack(embs)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X

def _author_prototypes(model, refs_json_path, device):
    data = json.loads(Path(refs_json_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        refs = {str(k): list(v) for k, v in data.items()}
    elif isinstance(data, list):
        if not data: raise ValueError("refs_json is empty.")
        # infer field names
        sample = data[0]
        text_key = "text" if "text" in sample else ("excerpt" if "excerpt" in sample else None)
        author_key = "author" if "author" in sample else ("name" if "name" in sample else None)
        if not text_key or not author_key:
            raise ValueError("Each item must have 'author' and 'text' (or 'excerpt'/'name').")
        refs_dd = defaultdict(list)
        for r in data:
            refs_dd[str(r[author_key])].append(str(r[text_key]))
        refs = dict(refs_dd)
    else:
        raise ValueError("refs_json must be dict or list of {author, text} objects.")

    names, proto_vecs = [], []
    for name, texts in refs.items():
        X = _embed_texts(model, texts, device)
        proto_vecs.append(X.mean(axis=0))
        names.append(name)

    P = np.vstack(proto_vecs)
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    return names, P

def _predict_from_prototypes(X, proto_names, P):
    S = X @ P.T
    idx = S.argmax(axis=1)
    y_pred = np.array([proto_names[i] for i in idx], dtype=object)
    conf = S.max(axis=1)
    return y_pred, conf

# collect unique (text, author) from the pairs loader and embed
def _collect_unique_from_pairs(model, loader, device):
    all_texts, all_authors, all_embs = [], [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Collecting t-SNE points", unit="batch", leave=False):
            # support both (s1,s2,y) and (s1,s2,y,a1,a2)
            if len(batch) == 3:
                s1, s2, _ = batch
                a1 = ["?"] * len(s1)
                a2 = ["?"] * len(s2)
            else:
                s1, s2, _, a1, a2 = batch

            e1 = model._embed(list(s1)).to("cpu").numpy()
            e2 = model._embed(list(s2)).to("cpu").numpy()
            all_texts.extend(list(s1) + list(s2))
            all_authors.extend([str(x) for x in a1] + [str(x) for x in a2])
            all_embs.append(np.vstack([e1, e2]))

    X = np.vstack(all_embs)
    texts = np.array(all_texts, dtype=object)
    authors = np.array(all_authors, dtype=object)

    # dedupe by text so each excerpt is one point
    _, keep_idx = np.unique(texts, return_index=True)
    X = X[keep_idx]
    authors = authors[keep_idx]
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X, authors

# plotting
def tsne_authors_plot(X, y_true, *, y_pred=None, P=None, proto_names=None, out_path="eval_plots/tsne_all_authors.png"):
    X = np.asarray(X); y_true = np.asarray(y_true, dtype=object)
    if P is not None:
        Z_all = TSNE(n_components=2, init="pca",
                     perplexity=min(50, max(5, len(X)//3)),
                     random_state=42).fit_transform(np.vstack([X, P]))
        Z, ZP = Z_all[:len(X)], Z_all[len(X):]
    else:
        Z = TSNE(n_components=2, init="pca",
                 perplexity=min(50, max(5, len(X)//3)),
                 random_state=42).fit_transform(X)
        ZP = None

    authors = np.unique(y_true)
    cmap = plt.get_cmap("tab20", len(authors))
    color_map = {a: cmap(i) for i, a in enumerate(authors)}
    point_colors = [color_map[a] for a in y_true]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(Z[:, 0], Z[:, 1], s=10, c=point_colors, alpha=0.75, linewidths=0)

    # misclassified
    if y_pred is not None:
        y_pred = np.asarray(y_pred, dtype=object)
        wrong = (y_true != y_pred)
        if wrong.any():
            ax.scatter(Z[wrong,0], Z[wrong,1], s=18, facecolors="none", edgecolors="k",
                       linewidths=0.8, label="misclassified")

    # prototypes
    if ZP is not None:
        ax.scatter(ZP[:,0], ZP[:,1], marker="*", s=220, edgecolor="k", linewidths=0.8,
                   facecolor="yellow", label="prototype")

    # legend swatches
    for a in authors:
        ax.scatter([], [], c=[color_map[a]], s=30, label=str(a))
    ax.legend(loc="best", fontsize=8, markerscale=1.2, frameon=False)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("t-SNE of Excerpt Embeddings\nColor: true author • misclassified • ★ prototypes")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300); plt.close(fig)

# ------------------------------ evaluate ----------------------------
def evaluate(
    ckpt_path,
    pairs_json,
    batch_size=64,
    device=None,
    plots=False,
    refs_json=None,            
    tsne_out="eval_plots/tsne_all_authors.png",
):
    # device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    device = torch.device(device)

    # model
    state = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
    weights = state.get("model_state_dict") or state.get("model") or state
    if weights is None:
        raise KeyError("No weights found in checkpoint (expected 'model_state_dict' or 'model').")

    model = SiameseModel(
        encoder_name="all-MiniLM-L6-v2",
        proj_dim=256, mlp_hidden=512, dropout=0.1, init_temp=10.0, device=device
    )
    model.load_state_dict(weights)
    model.to(device).eval()

    # classification metrics from pairs
    loader = _make_loader(pairs_json, batch_size)
    all_probs, all_preds, all_labels = [], [], []

    with torch.inference_mode():
        for batch_data in tqdm(loader, desc="Evaluating", unit="batch", leave=False):
            s1, s2, y = batch_data[:3]            # works for 3- or 5-tuple
            y = y.to(device)
            logits = model(s1, s2)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs > 0.5).int().cpu().numpy())
            all_labels.extend(y.int().cpu().numpy())

    labels = np.asarray(all_labels)
    preds  = np.asarray(all_preds)
    probs  = np.asarray(all_probs)

    metrics = {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "auc":       roc_auc_score(labels, probs) if len(np.unique(labels)) == 2 else None,
    }

    # t-SNE 
    if plots:
        # ---- use pairs split directly; show misclassified if refs are given ----
        X, y_true = _collect_unique_from_pairs(model, _make_loader(pairs_json, batch_size), device)
        y_pred = None; P = None; proto_names = None
        if refs_json:  # enable prototypes + misclassified from pairs
            proto_names, P = _author_prototypes(model, refs_json, device)
            y_pred, _ = _predict_from_prototypes(X, proto_names, P)
        tsne_authors_plot(X, y_true, y_pred=y_pred, P=P, proto_names=proto_names, out_path=tsne_out)

    return metrics
