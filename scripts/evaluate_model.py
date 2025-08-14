import os
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import joblib
import json
import matplotlib.pyplot as plt
import re


import torch
import torch.nn.functional as F
import random
from pathlib import Path
from models.sbert import SiameseModel
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import (
    INFERENCE_FILE, EVOLVE_DIR, PIC_DIR, EXCERPT_FILE, TRAIN_DATA,
    TFIDF_OUTPUT, STYLO_OUTPUT, W2V_OUTPUT, LSTM_OUTPUT, SBERT_OUTPUT,
)
from models.lstm import load_artifacts, TextDataset, pad_collate, encode_texts

# --------- registry ---------
SEED = 42

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
        "type": "sbert",        
        "artifacts": SBERT_OUTPUT,
    },

}

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

    # fixed author order and colors
    fixed_authors = [
        "Charlotte Brontë",
        "Edith Wharton",
        "George Eliot",
        "Henry James",
        "Virginia Woolf"
    ]
    fixed_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    color_map = {a: c for a, c in zip(fixed_authors, fixed_colors)}

    colors = [color_map[a] for a in labels]

    plt.figure(figsize=(7, 6))
    for author in fixed_authors:
        mask = [a == author for a in labels]
        plt.scatter(
            Y[mask, 0], Y[mask, 1],
            c=color_map[author], s=10, alpha=0.85, label=author
        )

    plt.legend(loc="best", fontsize=8, frameon=True)
    plt.title(title)
    plt.tight_layout()
    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


# --------- adapters ---------
def _eval_sklearn_prob(spec, excerpts):
    art = spec["artifacts"]

    # prefer a full pipeline that accepts raw text
    pipe_path = _find_first(art, ["pipeline.joblib", "pipe.joblib"])
    if pipe_path:
        clf = joblib.load(pipe_path)
        prob = clf.predict_proba(excerpts) if hasattr(clf, "predict_proba") \
               else _decision_to_prob(clf.decision_function(excerpts))
        classes = getattr(clf, "classes_", None)
        if classes is None and hasattr(clf, "steps"):
            classes = getattr(clf.steps[-1][1], "classes_", None)
        classes = classes.tolist() if classes is not None else list(range(prob.shape[1]))
    else:
        # fallback: separate vectorizer (+ optional scaler) + classifier
        clf_path = _find_first(art, spec.get("clf_keys", []))
        vec_path = _find_first(art, spec.get("vec_keys", []))
        assert clf_path, f"No classifier joblib found in {art}"
        assert vec_path, f"No vectorizer joblib found in {art} "

        clf = joblib.load(clf_path)
        vectorizer = joblib.load(vec_path)

        X = vectorizer.transform(excerpts)  # -> 2D feature matrix

        scaler_path = _find_first(art, spec.get("scaler_keys", []))
        if scaler_path:
            scaler = joblib.load(scaler_path)
            # handle sparse vs dense
            X = scaler.transform(X.toarray() if hasattr(X, "toarray") else X)

        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X)
        elif hasattr(clf, "decision_function"):
            prob = _decision_to_prob(clf.decision_function(X))
        else:
            raise RuntimeError("Classifier has neither predict_proba nor decision_function")

        classes = getattr(clf, "classes_", list(range(prob.shape[1])))

    # compact embedding source for t‑SNE 
    k = 2 if prob.shape[1] <= 2 else min(50, prob.shape[1] - 1)
    embed = TruncatedSVD(n_components=k, random_state=SEED).fit_transform(prob).astype(np.float32, copy=False)
    return classes, prob, embed

def _decision_to_prob(scores):
    scores = np.asarray(scores)
    if scores.ndim == 1:  # binary: shape (n,)
        scores = np.stack([-scores, scores], axis=1)
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def _eval_lstm(spec, excerpts):
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

    # fresh embedding
    embed = encode_texts(model, excerpts, stoi, classes, cfg).astype(np.float32, copy=False)

    return classes, prob, embed

# pair-scoring using trained MLP on pre-encoded embeddings + cosine kNN fusion
def _eval_sbert_pairs(
    spec, excerpts,
    agg="topm-sum", topk_per_author=150,
    batch_q=64, batch_r=512, ref_chunk_for_mlp=4096,
    tau=0.3, lam_prior=0.0, cap_per_author=300,
    alpha=0.2,                 # <-- fusion weight: 0=only SBERT, 1=only kNN
    knn_k=25                   # <-- number of neighbors for kNN vote
):

    art = Path(spec["artifacts"])
    ckpt = (art / "best.pt") if (art / "best.pt").exists() else (art / "last.pt")
    assert ckpt.exists(), f"No SBERT checkpoint in {art}"

    # ----- load weights -----
    def _load_sbert_weights(model, path):
        state = torch.load(str(path), map_location="cpu")
        raw = state.get("state_dict", state)
        def strip(k): return re.sub(r"^(module\.|model\.)", "", k)
        mapped = {strip(k): v for k, v in raw.items()}
        remap = {}
        for k, v in mapped.items():
            kk = k
            if kk.startswith("head."): kk = "scorer." + kk[len("head."):]
            if kk.startswith("mlp."):  kk = "scorer." + kk[len("mlp."):]
            remap[kk] = v
        model.load_state_dict(remap, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseModel(
        encoder_name="all-MiniLM-L6-v2",
        proj_dim=256, mlp_hidden=512, dropout=0.2, init_temp=10.0, device=device
    ).to(device)
    setattr(model, "_device", device)
    _load_sbert_weights(model, ckpt)
    model.eval()  # disable dropout in projector

    @torch.inference_mode()
    def _proj_encode(texts, bs):
        zs = []
        for i in range(0, len(texts), bs):
            e = model._embed_train(texts[i:i+bs])          # same path as training
            z = F.normalize(model.proj(e), p=2, dim=-1)    # L2-normalized
            zs.append(z.detach().cpu())
        return torch.cat(zs, dim=0)                        # CPU (N, D)

    # --- split & refs ---
    df = pd.read_csv(EXCERPT_FILE).reset_index(drop=True)
    with open(TRAIN_DATA, "r", encoding="utf-8") as f:
        split = json.load(f)
    test_idx = sorted(int(i) for i in split["test_index"])
    mask = np.zeros(len(df), dtype=bool); mask[test_idx] = True
    train_df = df.loc[~mask, ["excerpt", "author"]].reset_index(drop=True)

    authors = sorted(train_df["author"].unique().tolist())
    refs_by_author_texts = {a: train_df.loc[train_df["author"] == a, "excerpt"].tolist()
                            for a in authors}
    if cap_per_author:
        for a, lst in refs_by_author_texts.items():
            if len(lst) > cap_per_author:
                refs_by_author_texts[a] = random.sample(lst, cap_per_author)

    # pre-encode refs (projected, L2 normed)
    refs_by_author_Z = {a: _proj_encode(t, batch_r) if t else torch.empty(0, 256)
                        for a, t in refs_by_author_texts.items()}

    # centroids (for optional prior)
    centroids = []
    for a in authors:
        Zr = refs_by_author_Z[a].numpy()
        if Zr.shape[0] == 0:
            centroids.append(np.zeros((256,), dtype=np.float32))
        else:
            c = Zr.mean(axis=0).astype(np.float32, copy=False)
            c /= (np.linalg.norm(c) + 1e-12)
            centroids.append(c)
    C = np.stack(centroids, axis=0)  # (K, D)

    # encode queries once
    Zq_all = _proj_encode(excerpts, batch_q)  # CPU, L2-normalized
    embed = Zq_all.numpy().astype("float32", copy=False)

    # ====== build flat ref bank for kNN vote ======
    if alpha > 0.0:  # only if we actually use kNN
        ref_blocks = []
        ref_labels = []
        for a_idx, a in enumerate(authors):
            Zr = refs_by_author_Z[a]
            if Zr.numel() > 0:
                ref_blocks.append(Zr)                    # CPU tensor
                ref_labels.extend([a_idx] * Zr.shape[0])
        if ref_blocks:
            ALL_R = torch.cat(ref_blocks, dim=0)         # (Nr_total, D), CPU, normalized
            REF_L = np.asarray(ref_labels, dtype=np.int64)
        else:
            ALL_R = torch.empty(0, Zq_all.shape[1], dtype=torch.float32)
            REF_L = np.empty((0,), dtype=np.int64)

    @torch.inference_mode()
    def _score_block_on_embeddings(Zq_cpu, Zr_cpu):
        dev = model._device
        bq = int(Zq_cpu.shape[0])
        chunks = []
        for j in range(0, int(Zr_cpu.shape[0]), ref_chunk_for_mlp):
            Zr = Zr_cpu[j:j+ref_chunk_for_mlp].to(dev, non_blocking=True)   # (br, D)
            br = int(Zr.shape[0])
            if br == 0: continue
            Zq = Zq_cpu.to(dev, non_blocking=True)                           # (bq, D)
            zq = Zq.unsqueeze(1)                                             # (bq,1,D)
            zr = Zr.unsqueeze(0)                                             # (1,br,D)
            feats = torch.cat([zq * zr, (zq - zr).abs()], dim=-1).view(bq*br, -1)
            # calibrated LOGITS
            logits = model.scorer(feats).squeeze(-1) / model.temp
            logits = model.logit_scale * logits + model.logit_bias           # (bq*br,)
            L = logits.view(bq, br).cpu()                                    # (bq, br) CPU
            chunks.append(L)
            del Zr, Zq, zq, zr, feats, logits, L
            torch.cuda.empty_cache()
        if not chunks:
            return torch.empty(bq, 0, dtype=torch.float32)
        return torch.cat(chunks, dim=1)  # (bq, Nr_total) CPU

    N, K = len(excerpts), len(authors)
    probs = np.zeros((N, K), dtype="float32")

    print(f"[sbert-eval] agg={agg} topk_per_author={topk_per_author} alpha={alpha} knn_k={knn_k}")

    for qi in range(0, N, batch_q):
        Zq = Zq_all[qi:qi+batch_q]       # (bq, D) CPU, normalized
        bq = int(Zq.shape[0])
        S_tensors = []                   # (bq,) per author

        # ----- SBERT pair MLP per author (aggregate LOGITS) -----
        for a in authors:
            Zr = refs_by_author_Z[a]
            if Zr.numel() == 0:
                S_tensors.append(torch.zeros(bq, dtype=torch.float32))
                continue

            L_pairs = _score_block_on_embeddings(Zq, Zr)     # (bq, Nr) LOGITS
            if topk_per_author and L_pairs.shape[1] > topk_per_author:
                k = min(topk_per_author, L_pairs.shape[1])
                L_pairs = torch.topk(L_pairs, k=k, dim=1).values  # (bq, k)

            if agg == "topm-sum":
                m = min(10, L_pairs.shape[1])  # try 5/10/20
                agg_scores = torch.topk(L_pairs, k=m, dim=1).values.sum(dim=1)
            elif agg == "max":
                agg_scores = L_pairs.max(dim=1).values
            elif agg == "mean":
                agg_scores = L_pairs.mean(dim=1)
            else:
                raise ValueError("agg must be one of {'topm-sum','max','mean'}")

            S_tensors.append(agg_scores)

        # stack author scores → (bq, K) torch
        S = torch.stack(S_tensors, dim=1)  # (bq, K)

        # ----- cosine kNN vote on the same normalized embeddings -----
        if alpha > 0.0 and ALL_R.shape[0] > 0:
            # cosine since both are L2-normalized
            sim = Zq @ ALL_R.T                        # (bq, Nr_total)
            kk = min(knn_k, sim.shape[1])
            vals, idx = torch.topk(sim, k=kk, dim=1)
            # similarity-weighted votes per author
            w = torch.clamp(vals, min=0.0) ** 2       # nonneg, emphasize close hits
            knn_scores = torch.zeros(bq, K, dtype=torch.float32)
            idx_np = idx.cpu().numpy()
            w_np = w.cpu().numpy()
            for i in range(bq):
                np.add.at(knn_scores[i].numpy(), REF_L[idx_np[i]], w_np[i])
            # normalize per row to [0,1] range
            row_sum = knn_scores.sum(dim=1, keepdim=True).clamp_min(1e-6)
            knn_scores = knn_scores / row_sum         # (bq, K)
        else:
            knn_scores = torch.zeros_like(S)

        # ----- optional centroid prior -----
        if lam_prior > 0.0:
            Zq_np = Zq.numpy()
            cos = Zq_np @ C.T
            prior = 0.5 * (cos + 1.0)                 # [0,1]
            prior = torch.from_numpy(prior).to(S.dtype)
        else:
            prior = torch.zeros_like(S)

        # ----- fuse signals (z-score S only if mixing scales) -----
        # Bring S roughly onto 0-mean, unit-std per query (argmax-invariant by itself, but helpful for fusion).
        Sz = (S - S.mean(dim=1, keepdim=True)) / (S.std(dim=1, keepdim=True) + 1e-6)
        Pf = (1.0 - alpha) * Sz + alpha * knn_scores + lam_prior * prior

        # softmax over authors (tau doesn't change argmax for tau>0)
        Pf_np = Pf.cpu().numpy().astype("float32", copy=False)
        Zs = (Pf_np / tau) - (Pf_np / tau).max(axis=1, keepdims=True)
        e = np.exp(Zs, dtype=np.float64)
        P = e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)
        probs[qi:qi+bq, :] = P.astype("float32")

    classes = authors
    return classes, probs, embed


# --------- main ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="tfidf,stylometry,word2vec,lstm,sbert")
    args = ap.parse_args()

    df = _load_unseen()
    excerpts = df["excerpt"].tolist()
    gold = df["author"].tolist()
    runs_summary = []

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
            classes, prob, embed = _eval_sbert_pairs(spec, excerpts)
        else:
            classes, prob, embed = _eval_sklearn_prob(spec, excerpts)

        order = np.argsort(-prob, axis=1)
        topk = [[classes[j] for j in row[:3]] for row in order]
        top1 = [row[0] for row in topk]
        top1_acc = float(np.mean([g == p for g, p in zip(gold, top1)]))
        top2_acc = float(_topk_hits(gold, topk, k=2))

        print(f" Top-1 acc: {top1_acc:.4f}")
        print(f" Top-2 acc: {top2_acc:.4f}")

        # build per-example frame for breakdowns (to aggregate)
        per_example = pd.DataFrame({
            "author": gold,
            "book": df["book"].tolist(),
            "bucket": df["bucket"].tolist(),
            "pred_top1": top1,
        })
        per_example["hit_top1"] = per_example["author"] == per_example["pred_top1"]
        per_example["hit_top2"] = [g in c[:2] for g, c in zip(gold, topk)]

        # aggregate to the exact JSON shape you want
        by_author = (
            per_example.groupby("author")[["hit_top1", "hit_top2"]]
            .agg(["count", "mean"])
        )
        # flatten and cast
        by_author = {
            a: {"n": int(row[("hit_top1", "count")]),
                "top1_acc": float(row[("hit_top1", "mean")]),
                "top2_acc": float(row[("hit_top2", "mean")])}
            for a, row in by_author.iterrows()
        }

        by_book = (
            per_example.groupby("book")[["hit_top1", "hit_top2"]]
            .agg(["count", "mean"])
        )
        by_book = {
            b: {"n": int(row[("hit_top1", "count")]),
                "top1_acc": float(row[("hit_top1", "mean")]),
                "top2_acc": float(row[("hit_top2", "mean")])}
            for b, row in by_book.iterrows()
        }

        by_bucket = (
            per_example.groupby("bucket")[["hit_top1", "hit_top2"]]
            .agg(["count", "mean"])
        )
        by_bucket = {
            k: {"n": int(row[("hit_top1", "count")]),
                "top1_acc": float(row[("hit_top1", "mean")]),
                "top2_acc": float(row[("hit_top2", "mean")])}
            for k, row in by_bucket.iterrows()
        }

        runs_summary.append({
            "model": name,
            "overall": {"n": int(len(gold)), "top1_acc": float(top1_acc), "top2_acc": float(top2_acc)},
            "by_author": by_author,
            "by_book": by_book,
            "by_bucket": by_bucket,
        })

        out_png = os.path.join(PIC_DIR, f"{name}-tsne.png")
        # final guard to ensure lengths match (they should)
        if embed.shape[0] != len(gold):
            # fallback: derive from prob so it always matches
            k = 2 if prob.shape[1] <= 2 else min(50, prob.shape[1] - 1)
            embed = TruncatedSVD(n_components=k, random_state=SEED).fit_transform(prob).astype(np.float32, copy=False)
        _safe_tsne(embed, gold, title=f"{name} — t-SNE (unseen)", out_png=out_png)
        print(f" Saved t-SNE to {out_png}")
    
    # write exactly the minimal JSON you asked for
    metrics_path = os.path.join(EVOLVE_DIR,  f"{name}-metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(runs_summary, f, indent=2, ensure_ascii=False)
    print(f" Wrote JSON metrics to {metrics_path}")


if __name__ == "__main__":
    main()