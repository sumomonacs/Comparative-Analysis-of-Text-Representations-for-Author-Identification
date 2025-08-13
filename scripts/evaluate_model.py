import os
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import joblib
import json
import matplotlib.pyplot as plt

import torch
import random
from pathlib import Path
from models.sbert import SiameseModel

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import (
    INFERENCE_FILE, EVOLVE_DIR, PIC_DIR, EXCERPT_FILE, TRAIN_DATA,
    TFIDF_OUTPUT, STYLO_OUTPUT, W2V_OUTPUT, LSTM_OUTPUT, SBERT_OUTPUT,
)

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

    # compact embedding source for t‑SNE (always length-matched)
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

def _load_sbert_weights(model, ckpt_path):
    """
    Load SBERT weights from a checkpoint that stores under 'state_dict'
    (your training code), with common prefix fixes. Fails fast if keys
    are missing so you don't silently evaluate an untrained model.
    """
    import re, torch
    try:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location="cpu")

    raw = state.get("state_dict", state)   # <-- IMPORTANT: use 'state_dict'
    # strip common wrappers/prefixes
    def _strip(k):
        k = re.sub(r"^(module\.|model\.)", "", k)
        return k
    mapped = { _strip(k): v for k, v in raw.items() }

    # Optional: remap alternative head names if your train script used them
    remapped = {}
    for k, v in mapped.items():
        kk = k
        # e.g., if training named the head "head.*" or "mlp.*"
        if kk.startswith("head."):
            kk = "scorer." + kk[len("head."):]
        if kk.startswith("mlp."):
            kk = "scorer." + kk[len("mlp."):]
        remapped[kk] = v

    # Load and *fail fast* if something doesn't match (so you notice)
    ret = model.load_state_dict(remapped, strict=True)
    # If you prefer non-fatal, set strict=False but print:
    # ret = model.load_state_dict(remapped, strict=False)
    # print("[SBERT DEBUG] missing_keys:", ret.missing_keys)
    # print("[SBERT DEBUG] unexpected_keys:", ret.unexpected_keys)
    return ret


def _eval_sbert_pairs(spec, excerpts, agg="mean", topk_per_author=20,
                      batch_q=32, batch_r=512, ref_chunk_for_mlp=4096):
    """
    Pair-scoring using the trained MLP but WITHOUT re-encoding text per pair.

    - Pre-encode TRAIN refs once via model._embed_train -> proj -> L2 norm.
    - Pre-encode queries in batches the same way.
    - Score with model.scorer on features [zq*zr, |zq-zr|] in chunks.
    - Aggregate per author (probability-level), top-k filter, softmax across authors.

    Args:
      agg: "mean" or "max"
      topk_per_author: keep top-k strongest ref probs per author before aggregation
      batch_q: query batch size for embedding
      batch_r: reference batch size for embedding
      ref_chunk_for_mlp: how many ref vectors to score at once per query batch (MLP chunk)
    """
    import json, numpy as np, torch, pandas as pd
    from scripts.config import EXCERPT_FILE, TRAIN_DATA
    import torch.nn.functional as F

    art = Path(spec["artifacts"])
    ckpt = art / "best.pt"
    if not ckpt.exists():
        ckpt = art / "last.pt"
    assert ckpt.exists(), f"No SBERT checkpoint in {art}"

    # ---- load checkpoint ----
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

    art = Path(spec["artifacts"])
    ckpt = (art / "best.pt") if (art / "best.pt").exists() else (art / "last.pt")
    assert ckpt.exists(), f"No SBERT checkpoint in {art}"

    model = SiameseModel(
        encoder_name="all-MiniLM-L6-v2",
        proj_dim=256, mlp_hidden=512, dropout=0.1, init_temp=10.0, device=device
    ).to(device)

    # >>> replace your old load_state_dict code with:
    _load_sbert_weights(model, ckpt)
    model.eval()

    # ---------- helpers: encode via the *training* path ----------
    @torch.inference_mode()
    def _proj_encode(texts, bs):
        zs = []
        for i in range(0, len(texts), bs):
            e = model._embed_train(texts[i:i+bs])              # (B, hidden)
            z = F.normalize(model.proj(e), p=2, dim=-1)        # (B, D)
            zs.append(z.detach().cpu())
        return torch.cat(zs, dim=0)                            # (N, D) on CPU

    # ---------- load split & bucket refs by author ----------
    df = pd.read_csv(EXCERPT_FILE).reset_index(drop=True)
    with open(TRAIN_DATA, "r", encoding="utf-8") as f:
        split = json.load(f)
    test_idx = sorted(int(i) for i in split["test_index"])
    mask = np.zeros(len(df), dtype=bool); mask[test_idx] = True
    train_df = df.loc[~mask, ["excerpt", "author"]].reset_index(drop=True)

    authors = sorted(train_df["author"].unique().tolist())
    refs_by_author_texts = {a: train_df.loc[train_df["author"] == a, "excerpt"].tolist() for a in authors}

    # ---------- pre-encode all refs once (CPU tensors) ----------
    refs_by_author_Z = {}
    for a, texts in refs_by_author_texts.items():
        if texts:
            Z = _proj_encode(texts, batch_r)   # (Nr, D) on CPU
        else:
            Z = torch.empty(0, 256)            # empty fallback
        refs_by_author_Z[a] = Z
    
    cap_per_author = 300
    for a, lst in refs_by_author_texts.items():
        if len(lst) > cap_per_author:
            refs_by_author_texts[a] = random.sample(lst, cap_per_author)


    # ---------- main scoring ----------
    N = len(excerpts)
    K = len(authors)
    probs = np.zeros((N, K), dtype="float32")

    @torch.inference_mode()
    def _score_block_on_embeddings(Zq_cpu, Zr_cpu):
        """
        Zq_cpu: (Bq, D) CPU
        Zr_cpu: (Br, D) CPU
        returns (Bq, Br) probabilities
        """
        # move to device in manageable chunks
        Bq = Zq_cpu.shape[0]
        out = torch.empty(Bq, 0, dtype=torch.float32, device="cpu")
        for j in range(0, Zr_cpu.shape[0], ref_chunk_for_mlp):
            Zr = Zr_cpu[j:j+ref_chunk_for_mlp].to(device, non_blocking=True)     # (br, D)
            Zq = Zq_cpu.to(device, non_blocking=True)                             # (bq, D)

            # build features for all pairs in this chunk efficiently
            # broadcast to (bq, br, D)
            zq = Zq.unsqueeze(1)                  # (bq, 1, D)
            zr = Zr.unsqueeze(0)                  # (1, br, D)
            feats = torch.cat([zq * zr, (zq - zr).abs()], dim=-1)   # (bq, br, 2D)
            feats = feats.reshape(-1, feats.shape[-1])              # (bq*br, 2D)

            logits = model.scorer(feats).squeeze(-1) * model.temp   # (bq*br,)
            probs_chunk = torch.sigmoid(logits).reshape(Zq.shape[0], Zr.shape[0])  # (bq, br)

            out = torch.cat([out, probs_chunk.detach().cpu()], dim=1)
            del Zr, Zq, zq, zr, feats, logits, probs_chunk
            torch.cuda.empty_cache()
        return out  # CPU (bq, Br_total)

    # encode all queries (CPU tensors)
    Zq_all = _proj_encode(excerpts, batch_q)   # (N, D) CPU

    # for t‑SNE later: keep a copy of Zq (numpy)
    embed = Zq_all.numpy().astype("float32", copy=False)

    # loop over query batches
    for qi in range(0, N, batch_q):
        Zq = Zq_all[qi:qi+batch_q]   # CPU (bq, D)
        bq = Zq.shape[0]
        author_scores = []

        for a in authors:
            Zr = refs_by_author_Z[a]            # CPU (Nr, D)
            if Zr.numel() == 0:
                author_scores.append(np.full((bq,), 0.0, dtype="float32"))
                continue

            # (bq, Nr) probabilities
            P_pairs = _score_block_on_embeddings(Zq, Zr)

            # top‑k filter per row
            if topk_per_author is not None and P_pairs.shape[1] > topk_per_author:
                idx = np.argpartition(-P_pairs.numpy(), kth=topk_per_author-1, axis=1)[:, :topk_per_author]
                row_idx = np.arange(bq)[:, None]
                P_pairs = P_pairs[row_idx, idx]

            # aggregate
            if agg == "max":
                agg_scores = P_pairs.max(dim=1).values.numpy()
            else:
                agg_scores = P_pairs.mean(dim=1).numpy()

            author_scores.append(agg_scores.astype("float32"))

        # stack (bq, K) and softmax across authors with a mild temperature
        S = np.stack(author_scores, axis=1)             # (bq, K)
        tau = 0.3
        Zs = (S / tau) - (S / tau).max(axis=1, keepdims=True)
        e = np.exp(Zs)
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
            classes, prob, embed = _eval_sbert_pairs(
                spec, excerpts,
                agg="max",            # try max first; often best for OOD
                topk_per_author=80,   # 50–120 typically good; increase if each author has many refs
                batch_q=32,
                batch_r=512,
                ref_chunk_for_mlp=4096
            )
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