"""
    This file does deterministic SBERT training on contrastive pairs
"""

import os
import sys
import json
import random
from pathlib import Path

# determinism envs
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import PAIR_FILE, EXCERPT_FILE, TRAIN_DATA, SBERT_OUTPUT
from models.sbert import PairDataset, SiameseModel

# config / constants
SEED = 42

ENCODER_NAME = "all-MiniLM-L6-v2"
PROJ_DIM = 256
MLP_HIDDEN = 512
DROPOUT = 0.2
INIT_TEMP = 10.0

BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.0
FREEZE_LAYERS = 0  # freeze first N transformer layers

# ------------------- utils -------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def _load_frozen_indices(split_json):
    with open(split_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(int(i) for i in data["test_index"])

 # remove duplicate unordered pairs; return stable order
def _dedup_pairs(pairs):
    def key(d):
        a, b = d["text1"], d["text2"]
        return tuple(sorted((a, b)))
    seen = set()
    uniq = []
    for d in pairs:
        k = key(d)
        if k not in seen:
            uniq.append(d)
            seen.add(k)
    # stable sort for determinism
    uniq.sort(key=lambda d: (min(d["text1"], d["text2"]), max(d["text1"], d["text2"]), d["label"]))
    return uniq

# read excerpts and frozen test indices
def _split_pairs_frozen(pairs, excerpt_csv, split_json):
    df = pd.read_csv(excerpt_csv).reset_index(drop=True)
    test_idx = _load_frozen_indices(split_json)

    # texts from train/test sides 
    mask = np.zeros(len(df), dtype=bool)
    mask[test_idx] = True
    train_texts = set(df.loc[~mask, "excerpt"].tolist())
    test_texts  = set(df.loc[ mask, "excerpt"].tolist())

    # dedup before filtering
    pairs = _dedup_pairs(pairs)

    # assign by membership of both texts
    train_pairs = []
    val_pairs = []
    dropped = 0
    for d in pairs:
        t1 = d["text1"]; t2 = d["text2"]
        in_train = (t1 in train_texts) and (t2 in train_texts)
        in_test  = (t1 in test_texts)  and (t2 in test_texts)
        if in_train:
            train_pairs.append(d)
        elif in_test:
            val_pairs.append(d)
        else:
            dropped += 1  # cross-split

    return train_pairs, val_pairs, dropped

def _make_loader(source, batch_size, shuffle, seed):
    ds = PairDataset(source)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        worker_init_fn=None,
        generator=g,
        pin_memory=False,
    )

@torch.no_grad()
def _eval_metrics(model, loader, device):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y_true, y_prob = [], []
    for batch in loader:
        if len(batch) == 3:
            s1, s2, y = batch
        else:
            s1, s2, y, *_ = batch
        y = y.to(device)
        logits = model(s1, s2)
        prob = torch.sigmoid(logits)
        y_true.extend(y.cpu().numpy().tolist())
        y_prob.extend(prob.cpu().numpy().tolist())
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }

def main():
    os.makedirs(SBERT_OUTPUT, exist_ok=True)
    set_seed(SEED)

    # load pairs
    with open(PAIR_FILE, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    # frozen split for pairs
    train_pairs, val_pairs, dropped = _split_pairs_frozen(pairs, EXCERPT_FILE, TRAIN_DATA)
    print(f"Pairs: train={len(train_pairs)}  val={len(val_pairs)}  dropped_cross_split={dropped}")

    # basic sanity
    if len(train_pairs) == 0 or len(val_pairs) == 0:
        print("[WARN] One of the splits has zero pairs. Check your PAIR_FILE vs EXCERPT_FILE/indices.")

    # dataLoaders (deterministic)
    train_loader = _make_loader(train_pairs, BATCH_SIZE, shuffle=True,  seed=SEED)
    val_loader   = _make_loader(val_pairs,   BATCH_SIZE, shuffle=False, seed=SEED)

    # device
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    # model
    model = SiameseModel(
        encoder_name=ENCODER_NAME,
        proj_dim=PROJ_DIM,
        mlp_hidden=MLP_HIDDEN,
        dropout=DROPOUT,
        init_temp=INIT_TEMP,
        device=device,
    ).to(device)

    if FREEZE_LAYERS > 0:
        model.freeze_encoder_layers(FREEZE_LAYERS)

    # optim / loss
    loss_fn  = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # train
    best_val = float("inf")
    best_path = Path(SBERT_OUTPUT) / "best.pt"
    last_path = Path(SBERT_OUTPUT) / "last.pt"

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        # train
        model.train()
        total, running = 0, 0.0
        for batch in tqdm(train_loader, desc="train", leave=False):
            if len(batch) == 3:
                s1, s2, y = batch
            else:
                s1, s2, y, *_ = batch
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(s1, s2)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            running += loss.item() * y.size(0)
            total += y.size(0)
        tr_loss = running / max(total, 1)

        # val
        model.eval()
        total, running = 0, 0.0
        for batch in tqdm(val_loader, desc="val", leave=False):
            if len(batch) == 3:
                s1, s2, y = batch
            else:
                s1, s2, y, *_ = batch
            y = y.to(device)
            logits = model(s1, s2)
            loss = loss_fn(logits, y)
            running += loss.item() * y.size(0)
            total += y.size(0)
        va_loss = running / max(total, 1)
        metrics = _eval_metrics(model, val_loader, device)
        print(f" train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
              f"acc={metrics['acc']:.4f}  f1={metrics['macro_f1']:.4f}  auc={metrics['roc_auc']:.4f}")

        # save last (CPU tensors)
        torch.save({
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "epoch": epoch,
            "val_loss": va_loss,
            "metrics": metrics,
            "config": {
                "seed": SEED,
                "encoder_name": ENCODER_NAME,
                "proj_dim": PROJ_DIM,
                "mlp_hidden": MLP_HIDDEN,
                "dropout": DROPOUT,
                "init_temp": INIT_TEMP,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "freeze_layers": FREEZE_LAYERS,
                "split_file": TRAIN_DATA,
                "pairs_file": PAIR_FILE,
            }
        }, last_path)

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "val_loss": va_loss,
                "metrics": metrics,
            }, best_path)
            print(f"  ↳ new best saved to {best_path}")

    print(f"\nFinished. Best checkpoint: {best_path}")

if __name__ == "__main__":
    main()
