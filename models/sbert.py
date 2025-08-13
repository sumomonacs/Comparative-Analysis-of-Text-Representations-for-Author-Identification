
"""
    This file defines Sentence-BERT Siamese model + dataset utilities 

"""

import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

# dataset 
class PairDataset(Dataset):
    def __init__(self, pair_source):
        if isinstance(pair_source, (str, Path)):
            with Path(pair_source).open("r", encoding="utf-8") as f:
                data = json.load(f)
        elif hasattr(pair_source, "read"):
            data = json.load(pair_source)
        else:
            data = pair_source

        self.text1   = [d["text1"] for d in data]
        self.text2   = [d["text2"] for d in data]
        self.labels  = torch.tensor([d["label"] for d in data], dtype=torch.float32)
        self.author1 = [d.get("author1") for d in data]
        self.author2 = [d.get("author2") for d in data]
        self.has_authors = any(a is not None for a in self.author1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        if self.has_authors:
            return self.text1[idx], self.text2[idx], self.labels[idx], self.author1[idx], self.author2[idx]
        return self.text1[idx], self.text2[idx], self.labels[idx]

# Siamese Sentence-BERT model
class SiameseModel(nn.Module):
    """
    SBERT encoder (HF Transformer via sentence-transformers)
    -> projection head (nonlinear)
    -> pair scorer (concat[z1*z2, |z1-z2|]) -> 1 logit
    """
    def __init__(self, encoder_name = "all-MiniLM-L6-v2", proj_dim=256, mlp_hidden=512,
        dropout=0.2, init_temp=10.0, device= None):
        super().__init__()

        if device is None:
            device = (
                torch.device("cuda") if torch.cuda.is_available()
                else torch.device("mps") if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        self._device = torch.device(device)

        # sentence-transformers wrapper (tokenizer + encoder stack)
        self.encoder = SentenceTransformer(encoder_name, device=self._device)
        self.encoder.to(self._device)
        self._hf_model = self.encoder._first_module().auto_model.to(self._device)
        emb_dim = self.encoder.get_sentence_embedding_dimension()

        # projection head to a compact space
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim, bias=False),
        )

        # pair scorer
        self.scorer = nn.Sequential(
            nn.Linear(proj_dim * 2, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1, bias=False),
        )

        # learnable temperature to scale logits (useful with BCEWithLogitsLoss)
        self.temp = nn.Parameter(torch.tensor([init_temp], dtype=torch.float32))

    # ---- internals ----
    # tokenize with SBERT's tokenizer, run HF encoder with gradients, mean-pool with attention mask.
    def _embed_train(self, sentences):
        tok = self.encoder.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        out = self._hf_model(**tok, return_dict=True)  # gradients ON
        mask = tok["attention_mask"].unsqueeze(-1)     # (B, T, 1)
        emb  = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return emb  # (B, hidden)

    # forward / encode
    def forward(self, s1, s2):
        e1 = self._embed_train(list(s1))
        e2 = self._embed_train(list(s2))
        z1 = F.normalize(self.proj(e1), p=2, dim=-1)
        z2 = F.normalize(self.proj(e2), p=2, dim=-1)
        feats = torch.cat([z1 * z2, (z1 - z2).abs()], dim=-1)
        logits = self.scorer(feats).squeeze(-1) * self.temp
        return logits  # (B,)

    # inference-only embeddings 
    @torch.inference_mode()
    def encode(self, texts, batch_size=64):
        embs = []
        for i in range(0, len(texts), batch_size):
            embs.append(self._embed_train(texts[i:i+batch_size]))  # use same path as training
        e = torch.cat(embs, dim=0)
        z = F.normalize(self.proj(e), p=2, dim=-1)
        return z

    @property
    def embedding_dim(self):
        return int(self.encoder.get_sentence_embedding_dimension())

    #  freeze the first n Transformer layers in the HF encoder.
    def freeze_encoder_layers(self, n_layers):
        bert = self._hf_model
        if hasattr(bert, "encoder") and hasattr(bert.encoder, "layer"):
            for i, layer in enumerate(bert.encoder.layer):
                if i < n_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
