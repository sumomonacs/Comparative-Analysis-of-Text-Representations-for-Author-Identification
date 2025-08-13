"""
    This file does deterministic SBERT training on contrastive pairs
Usage :
python -m scripts.train_sbert \
  --encoder all-mpnet-base-v2 --proj-dim 256 --mlp-hidden 512 \
  --dropout 0.2 --init-temp 10 --batch-size 16 --epochs 3 \
  --lr 2e-5 --weight-decay 0.0 --freeze-layers 0 --seed 42
"""

import os
import sys
import json
import random
from pathlib import Path
from dataclasses import asdict, dataclass

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
import argparse
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import PAIR_FILE, EXCERPT_FILE, TRAIN_DATA, SBERT_OUTPUT
from models.sbert import PairDataset, SiameseModel

# ------------------- config -------------------
@dataclass
class CFG:
    seed: int = 42
    encoder_name: str = "all-MiniLM-L6-v2"
    proj_dim: int = 256
    mlp_hidden: int = 512
    dropout: float = 0.2
    init_temp: float = 10.0
    batch_size: int = 32
    epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.0
    freeze_layers: int = 0
    device: str | None = None  # "cuda" | "mps" | "cpu" | None=auto

    # paths (overridable)
    pair_file: str = PAIR_FILE
    excerpt_file: str = EXCERPT_FILE
    split_file: str = TRAIN_DATA
    out_dir: str = SBERT_OUTPUT


def parse_args():
    p = argparse.ArgumentParser(add_help=True)
    # model/opt
    p.add_argument("--encoder", dest="encoder_name", type=str, default=CFG.encoder_name)
    p.add_argument("--proj-dim", type=int, default=CFG.proj_dim)
    p.add_argument("--mlp-hidden", type=int, default=CFG.mlp_hidden)
    p.add_argument("--dropout", type=float, default=CFG.dropout)
    p.add_argument("--init-temp", type=float, default=CFG.init_temp)
    p.add_argument("--batch-size", type=int, default=CFG.batch_size)
    p.add_argument("--epochs", type=int, default=CFG.epochs)
    p.add_argument("--lr", type=float, default=CFG.lr)
    p.add_argument("--weight-decay", type=float, default=CFG.weight_decay)
    p.add_argument("--freeze-layers", type=int, default=CFG.freeze_layers)
    p.add_argument("--seed", type=int, default=CFG.seed)
    p.add_argument("--device", type=str, choices=["cuda", "mps", "cpu"], default=None)

    # io
    p.add_argument("--pair-file", type=str, default=CFG.pair_file)
    p.add_argument("--excerpt-file", type=str, default=CFG.excerpt_file)
    p.add_argument("--split-file", type=str, default=CFG.split_file)
    p.add_argument("--out-dir", type=str, default=CFG.out_dir)
    return p.parse_args()


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


def _dedup_pairs(pairs):
    # remove duplicate unordered pairs; return stable order
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
    uniq.sort(key=lambda d: (min(d["text1"], d["text2"]), max(d["text1"], d["text2"]), d["label"]))
    return uniq


def _split_pairs_frozen(pairs, excerpt_csv, split_json):
    df = pd.read_csv(excerpt_csv).reset_index(drop=True)
    test_idx = _load_frozen_indices(split_json)

    mask = np.zeros(len(df), dtype=bool)
    mask[test_idx] = True
    train_texts = set(df.loc[~mask, "excerpt"].tolist())
    test_texts  = set(df.loc[ mask, "excerpt"].tolist())

    pairs = _dedup_pairs(pairs)

    train_pairs, val_pairs, dropped = [], [], 0
    for d in pairs:
        t1, t2 = d["text1"], d["text2"]
        in_train = (t1 in train_texts) and (t2 in train_texts)
        in_test  = (t1 in test_texts)  and (t2 in test_texts)
        if in_train:
            train_pairs.append(d)
        elif in_test:
            val_pairs.append(d)
        else:
            dropped += 1
    return train_pairs, val_pairs, dropped


def _make_loader(source, batch_size, shuffle, seed):
    ds = PairDataset(source)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        drop_last=False, num_workers=0, worker_init_fn=None,
        generator=g, pin_memory=False,
    )


@torch.no_grad()
def _eval_metrics(model, loader, device):
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
    args = parse_args()
    cfg = CFG(
        seed=args.seed,
        encoder_name=args.encoder_name,
        proj_dim=args.proj_dim,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
        init_temp=args.init_temp,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        freeze_layers=args.freeze_layers,
        device=args.device,
        pair_file=args.pair_file,
        excerpt_file=args.excerpt_file,
        split_file=args.split_file,
        out_dir=args.out_dir,
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    # device (auto if None)
    if cfg.device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device(cfg.device)

    print(">> EFFECTIVE CONFIG:", {**asdict(cfg), "device": str(device)}, flush=True)

    # load pairs & split
    with open(cfg.pair_file, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    train_pairs, val_pairs, dropped = _split_pairs_frozen(pairs, cfg.excerpt_file, cfg.split_file)
    print(f"Pairs: train={len(train_pairs)}  val={len(val_pairs)}  dropped_cross_split={dropped}", flush=True)

    # loaders
    train_loader = _make_loader(train_pairs, cfg.batch_size, shuffle=True,  seed=cfg.seed)
    val_loader   = _make_loader(val_pairs,   cfg.batch_size, shuffle=False, seed=cfg.seed)

    # model
    model = SiameseModel(
        encoder_name=cfg.encoder_name,
        proj_dim=cfg.proj_dim,
        mlp_hidden=cfg.mlp_hidden,
        dropout=cfg.dropout,
        init_temp=cfg.init_temp,
        device=device,
    ).to(device)

    if cfg.freeze_layers > 0:
        model.freeze_encoder_layers(cfg.freeze_layers)

    # optim / loss
    loss_fn  = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # train
    best_val = float("inf")
    best_path = Path(cfg.out_dir) / "best.pt"
    last_path = Path(cfg.out_dir) / "last.pt"

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")

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
                **asdict(cfg),
                "device": str(device),
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
