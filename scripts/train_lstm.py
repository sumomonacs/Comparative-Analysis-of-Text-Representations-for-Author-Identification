"""
    This file train/evaluate a BiLSTM author classifier 
"""
import os
import json
import sys
import random
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import EXCERPT_FILE, TRAIN_DATA, LSTM_OUTPUT

# model pieces
from models.lstm import (
    LSTMClassifier, LSTMConfig,
    build_vocab, TextDataset, pad_collate,
    encode_texts, save_artifacts
)

# ---- config / output ----
CFG = LSTMConfig(emb_dim=200, hidden=256, layers=1, dropout=0.3,
                 pool="mean", max_len=256, epochs=8, batch_size=32,
                 lr=2e-3, weight_decay=0.0, seed=42)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------------------------- helpers ---------------------------
# load frozen split
def _load_frozen_test_indices(json_path: str):
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
    os.makedirs(LSTM_OUTPUT, exist_ok=True)
    set_seed(CFG.seed)

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
    stoi = build_vocab(Xtr, min_count=CFG.min_count,
                       lowercase=CFG.lowercase, alpha_only=CFG.alpha_only,
                       max_size=CFG.vocab_max_size)
    classes = sorted(list(set(ytr)))

    # datasets / loaders
    train_ds = TextDataset(Xtr, ytr, stoi, classes, CFG.lowercase, CFG.alpha_only, CFG.max_len)
    test_ds  = TextDataset(Xte, yte, stoi, classes, CFG.lowercase, CFG.alpha_only, CFG.max_len)
    train_dl = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,  collate_fn=pad_collate)
    test_dl  = DataLoader(test_ds,  batch_size=CFG.batch_size, shuffle=False, collate_fn=pad_collate)

    # model / optim
    model = LSTMClassifier(vocab_size=len(stoi), num_classes=len(classes),
                           emb_dim=CFG.emb_dim, hidden=CFG.hidden,
                           layers=CFG.layers, dropout=CFG.dropout, pool=CFG.pool).to(CFG.device)
    optim = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    crit  = nn.CrossEntropyLoss()

    # train
    def train_epoch():
        model.train(); tot, n = 0.0, 0
        for xb, yb, lengths in train_dl:
            xb, yb, lengths = xb.to(CFG.device), yb.to(CFG.device), lengths.to(CFG.device)
            optim.zero_grad()
            logits, _ = model(xb, lengths)
            loss = crit(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tot += loss.item() * xb.size(0); n += xb.size(0)
        return tot / max(n, 1)

    for e in range(1, CFG.epochs + 1):
        loss = train_epoch()
        print(f"[epoch {e}/{CFG.epochs}] loss={loss:.4f}")

    # ealuate
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb, lengths in test_dl:
            xb, yb, lengths = xb.to(CFG.device), yb.to(CFG.device), lengths.to(CFG.device)
            logits, _ = model(xb, lengths)
            pred = logits.argmax(-1)
            y_true.extend(yb.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    # map back to labels
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
    print(f"Macro‑F1  : {mf1:.4f}")
    print(rep)
    print("Confusion matrix:\n", cm)

    # save artifacts / outputs
    save_artifacts(LSTM_OUTPUT, model, stoi, classes, CFG)
    pd.DataFrame({"text": Xte, "author": test_df["author"], "pred": preds_labels}).to_csv(
        os.path.join(LSTM_OUTPUT, "preds.csv"), index=False
    )
    with open(os.path.join(LSTM_OUTPUT, "classification_report.txt"), "w") as f:
        f.write(rep)
    with open(os.path.join(LSTM_OUTPUT, "metrics.json"), "w") as f:
        json.dump({"accuracy": float(acc), "macro_f1": float(mf1), "split": split_mode}, f, indent=2)

    # export embeddings for t‑SNE
    Z = encode_texts(model, Xte, stoi, classes, CFG)
    np.save(os.path.join(LSTM_OUTPUT, "emb_test.npy"), Z.astype(np.float32))

    # save meta
    meta = {
        "data": EXCERPT_FILE,
        "split_file": TRAIN_DATA if os.path.exists(TRAIN_DATA) else None,
        "n_test": len(Xte),
        "cfg": asdict(CFG),
        "classes": classes,
    }
    
    with open(os.path.join(LSTM_OUTPUT, "meta.json"), "w") as f:
        json.dump(serialize_meta(meta), f, indent=2)

    print(f"\nSaved artifacts and metrics to: {LSTM_OUTPUT}")

if __name__ == "__main__":
    main()
