"""
    This file defines BiLSTM author model + utilities.
    - Tokenization, vocab, dataset/collate
    - BiLSTM encoder + small MLP head
    - Encode helper to get fixed-size embeddings for t-SNE
    - Save/load artifacts (state_dict + meta)
"""

from dataclasses import dataclass
from typing import Optional

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
nltk.download("punkt")

PAD, UNK = "<pad>", "<unk>"


# tokenization
def tok(text, lowercase=True, alpha_only= True):
    t = word_tokenize(text)
    if lowercase:
        t = [w.lower() for w in t]
    if alpha_only:
        t = [w for w in t if w.isalpha()]
    return t

def build_vocab(texts, min_count=2, lowercase=True, alpha_only=True, max_size=30000):
    cnt = Counter()
    for s in texts:
        cnt.update(tok(s, lowercase, alpha_only))
    words = [w for w, c in cnt.items() if c >= min_count]
    words = sorted(words, key=lambda w: (-cnt[w], w))  # freq desc, tie-break lexicographic
    if max_size is not None:
        words = words[:max_size]
    itos = [PAD, UNK] + words
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi

def encode_tokens(tokens, stoi, max_len):
    ids = [stoi.get(w, stoi[UNK]) for w in tokens]
    return ids[:max_len]


# dataset / collate
class TextDataset(Dataset):
    def __init__(self, texts, labels, stoi, classes=None, lowercase=True, alpha_only=True, max_len=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.stoi = stoi
        self.lowercase = lowercase
        self.alpha_only = alpha_only
        self.max_len = max_len
        self.classes = classes or sorted(list(set(labels)))
        self.y_map = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i: int):
        x = encode_tokens(tok(self.texts[i], self.lowercase, self.alpha_only), self.stoi, self.max_len)
        y = self.y_map[self.labels[i]]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def pad_collate(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    maxlen = int(lengths.max().item()) if len(lengths) else 1
    padded = torch.full((len(xs), maxlen), 0, dtype=torch.long)  # PAD=0
    for i, x in enumerate(xs):
        padded[i, :len(x)] = x
    return padded, torch.stack(ys), lengths

# model
class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, layers,
                 dropout, pad_idx=0, bidirectional=True):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden * (2 if bidirectional else 1)

    def forward(self, x, lengths, pool="mean"):
        mask = (x != 0).float()         # (B,T)
        emb = self.emb(x)               # (B,T,E)
        out, _ = self.lstm(emb)         # (B,T,H*dirs)
        out = self.drop(out)

        if pool == "mean":
            denom = torch.clamp(mask.sum(1, keepdim=True), min=1.0)
            z = (out * mask.unsqueeze(-1)).sum(1) / denom
        elif pool == "max":
            z = out.masked_fill(mask.unsqueeze(-1) == 0, float("-inf")).max(1).values
        elif pool == "last":
            idx = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(out.size(0), 1, out.size(2))
            z = out.gather(1, idx).squeeze(1)
        else:
            raise ValueError("pool must be mean|max|last")
        return z  # (B, D)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_dim=200, hidden=256, layers=1, dropout=0.3, pool="mean"):
        super().__init__()
        self.encoder = BiLSTMEncoder(vocab_size, emb_dim, hidden, layers, dropout, pad_idx=0, bidirectional=True)
        self.pool = pool
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, lengths):
        z = self.encoder(x, lengths, pool=self.pool)
        logits = self.mlp(z)
        return logits, z  # logits for classification, z for embeddings

# config 
@dataclass
class LSTMConfig:
    emb_dim: int = 200
    hidden: int = 256
    layers: int = 1
    dropout: float = 0.3
    pool: str = "mean"                 # "mean" | "max" | "last"
    max_len: int = 256
    min_count: int = 2
    vocab_max_size: Optional[int] = 30000
    lowercase: bool = True
    alpha_only: bool = True
    batch_size: int = 32
    epochs: int = 8
    lr: float = 2e-3
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

#  return fixed-size LSTM embeddings for given texts
def encode_texts(model, texts, stoi, classes, cfg):
    dummy_labels = [classes[0]] * len(texts)
    ds = TextDataset(texts, dummy_labels, stoi, classes=classes,
                     lowercase=cfg.lowercase, alpha_only=cfg.alpha_only, max_len=cfg.max_len)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)
    model.eval()
    Z = []
    with torch.no_grad():
        for xb, _, lengths in dl:
            xb, lengths = xb.to(cfg.device), lengths.to(cfg.device)
            _, z = model(xb, lengths)
            Z.append(z.cpu().numpy())
    return np.vstack(Z).astype(np.float32)

# save / load
def save_artifacts(out_dir: str, model, stoi, classes, cfg):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "lstm.pt"))
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"stoi": stoi, "classes": classes, "cfg": cfg}, f)

def load_artifacts(out_dir):
    with open(os.path.join(out_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stoi = meta["stoi"]; classes = meta["classes"]; cfg: LSTMConfig = meta["cfg"]
    model = LSTMClassifier(vocab_size=len(stoi), num_classes=len(classes),
                           emb_dim=cfg.emb_dim, hidden=cfg.hidden,
                           layers=cfg.layers, dropout=cfg.dropout, pool=cfg.pool)
    model.load_state_dict(torch.load(os.path.join(out_dir, "lstm.pt"), map_location="cpu"))
    model.eval()
    return model, stoi, classes, cfg
