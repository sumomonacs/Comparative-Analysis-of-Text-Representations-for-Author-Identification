"""
    This file train/evaluate a BiLSTM author classifier 
    Best hyperparameter:
    python -m scripts.train_lstm --emb-dim 300 --hidden 512 --epochs 8 --dropout 0.2 --pool "max"
"""
import os
import json
import sys
import random
from pathlib import Path
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import EXCERPT_FILE, TRAIN_DATA, LSTM_OUTPUT

from models.lstm import (
    LSTMClassifier, LSTMConfig,
    build_vocab, TextDataset, pad_collate,
    encode_texts, save_artifacts
)

# config / output
BASE_CFG = LSTMConfig(emb_dim=200, hidden=256, layers=1, dropout=0.3,
                      pool="mean", max_len=256, epochs=8, batch_size=32,
                      lr=2e-3, weight_decay=0.0, seed=42)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------------------------- helpers ---------------------------
def _load_frozen_test_indices(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(int(i) for i in data["test_index"])

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

# --------------------------- CLI ---------------------------
def _bool_flag(parser, name, default):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name.replace("-", "_"), action="store_true")
    group.add_argument(f"--no-{name}", dest=name.replace("-", "_"), action="store_false")
    parser.set_defaults(**{name.replace("-", "_"): default})

def parse_args():
    p = argparse.ArgumentParser(add_help=True)

    # tweakable training/model knobs
    p.add_argument("--emb-dim", type=int, default=BASE_CFG.emb_dim)
    p.add_argument("--hidden", type=int, default=BASE_CFG.hidden)
    p.add_argument("--layers", type=int, default=BASE_CFG.layers)
    p.add_argument("--dropout", type=float, default=BASE_CFG.dropout)
    p.add_argument("--pool", type=str, choices=["mean", "max", "last"], default=BASE_CFG.pool)
    p.add_argument("--max-len", type=int, default=BASE_CFG.max_len)

    p.add_argument("--epochs", type=int, default=BASE_CFG.epochs)
    p.add_argument("--batch-size", type=int, default=BASE_CFG.batch_size)
    p.add_argument("--lr", type=float, default=BASE_CFG.lr)
    p.add_argument("--weight-decay", type=float, default=BASE_CFG.weight_decay)
    p.add_argument("--seed", type=int, default=BASE_CFG.seed)

    # vocab/build flags 
    p.add_argument("--min-count", type=int, default=getattr(BASE_CFG, "min_count", 1))
    p.add_argument("--vocab-max-size", type=int, default=getattr(BASE_CFG, "vocab_max_size", 50000))
    _bool_flag(p, "lowercase", getattr(BASE_CFG, "lowercase", True))
    _bool_flag(p, "alpha-only", getattr(BASE_CFG, "alpha_only", True))

    # device 
    p.add_argument("--device", type=str, default=getattr(BASE_CFG, "device", "cuda" if torch.cuda.is_available() else "cpu"))

    return p.parse_args()

def _apply_overrides(cfg, args):
    return replace(
        cfg,
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        pool=args.pool,
        max_len=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        **{
            k: getattr(args, k)
            for k in ["min_count", "vocab_max_size", "lowercase", "alpha_only", "device"]
            if hasattr(cfg, k) and hasattr(args, k)
        }
    )


def main():
    args = parse_args()
    cfg = _apply_overrides(BASE_CFG, args)

    os.makedirs(LSTM_OUTPUT, exist_ok=True)
    set_seed(cfg.seed)

    # load data
    df = pd.read_csv(EXCERPT_FILE).reset_index(drop=True)
    assert {"excerpt", "author"}.issubset(df.columns), "CSV must contain 'excerpt' and 'author'."

    # split
    test_idx = _load_frozen_test_indices(TRAIN_DATA)
    mask = np.zeros(len(df), dtype=bool); mask[test_idx] = True
    train_df = df.loc[~mask].reset_index(drop=True)
    test_df  = df.loc[mask].reset_index(drop=True)
    split_mode = f"frozen({TRAIN_DATA})"

    Xtr = train_df["excerpt"].tolist(); ytr = train_df["author"].tolist()
    Xte = test_df["excerpt"].tolist();  yte = test_df["author"].tolist()

    # vocab from train only 
    stoi = build_vocab(Xtr, min_count=getattr(cfg, "min_count", 1),
                       lowercase=getattr(cfg, "lowercase", True), alpha_only=getattr(cfg, "alpha_only", True),
                       max_size=getattr(cfg, "vocab_max_size", 50000))
    classes = sorted(list(set(ytr)))

    # datasets / loaders
    train_ds = TextDataset(Xtr, ytr, stoi, classes, getattr(cfg, "lowercase", True),
                           getattr(cfg, "alpha_only", True), cfg.max_len)
    test_ds  = TextDataset(Xte, yte, stoi, classes, getattr(cfg, "lowercase", True),
                           getattr(cfg, "alpha_only", True), cfg.max_len)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  collate_fn=pad_collate)
    test_dl  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)

    model = LSTMClassifier(vocab_size=len(stoi), num_classes=len(classes),
                           emb_dim=cfg.emb_dim, hidden=cfg.hidden,
                           layers=cfg.layers, dropout=cfg.dropout, pool=cfg.pool).to(getattr(cfg, "device", "cpu"))
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit  = nn.CrossEntropyLoss()

    epochs = int(cfg.epochs)
    print(">>> EPOCHS:", epochs, flush=True)


    # train
    def train_epoch():
        model.train(); tot, n = 0.0, 0
        for xb, yb, lengths in train_dl:
            dev = getattr(cfg, "device", "cpu")
            xb, yb, lengths = xb.to(dev), yb.to(dev), lengths.to(dev)
            optim.zero_grad()
            logits, _ = model(xb, lengths)
            loss = crit(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tot += loss.item() * xb.size(0); n += xb.size(0)
        return tot / max(n, 1)

    for e in range(1, epochs + 1):
        loss = train_epoch()
        print(f"[epoch {e}/{epochs}] loss={loss:.4f}", flush=True)

    # evaluate
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb, lengths in test_dl:
            xb, yb, lengths = xb.to(getattr(cfg, "device", "cpu")), yb.to(getattr(cfg, "device", "cpu")), lengths.to(getattr(cfg, "device", "cpu"))
            logits, _ = model(xb, lengths)
            pred = logits.argmax(-1)
            y_true.extend(yb.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    id2label = {i: c for i, c in enumerate(classes)}
    preds_labels = [id2label[i] for i in y_pred]
    true_labels  = [id2label[i] for i in y_true]

    acc  = accuracy_score(true_labels, preds_labels)
    mf1  = f1_score(true_labels, preds_labels, average="macro")
    rep  = classification_report(true_labels, preds_labels, labels=classes)
    cm   = confusion_matrix(true_labels, preds_labels, labels=classes)

    print("\n--- BiLSTM Author Classifier ---")
    print(f"Split     : {split_mode}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Macro-F1  : {mf1:.4f}")
    print(rep)
    print("Confusion matrix:\n", cm)

    # save artifacts / outputs 
    save_artifacts(LSTM_OUTPUT, model, stoi, classes, cfg)
    pd.DataFrame({"text": Xte, "author": test_df["author"], "pred": preds_labels}).to_csv(
        os.path.join(LSTM_OUTPUT, "preds.csv"), index=False
    )
    with open(os.path.join(LSTM_OUTPUT, "classification_report.txt"), "w") as f:
        f.write(rep)
    with open(os.path.join(LSTM_OUTPUT, "metrics.json"), "w") as f:
        json.dump({"accuracy": float(acc), "macro_f1": float(mf1), "split": split_mode}, f, indent=2)

    # export embeddings for t-SNE
    Z = encode_texts(model, Xte, stoi, classes, cfg)
    np.save(os.path.join(LSTM_OUTPUT, "emb_test.npy"), Z.astype(np.float32))

    # save meta
    meta = {
        "data": EXCERPT_FILE,
        "split_file": TRAIN_DATA if os.path.exists(TRAIN_DATA) else None,
        "n_test": len(Xte),
        "cfg": asdict(cfg),
        "classes": classes,
    }
    with open(os.path.join(LSTM_OUTPUT, "meta.json"), "w") as f:
        json.dump(serialize_meta(meta), f, indent=2)

    print(f"\nSaved artifacts and metrics to: {LSTM_OUTPUT}")

if __name__ == "__main__":
    main()
