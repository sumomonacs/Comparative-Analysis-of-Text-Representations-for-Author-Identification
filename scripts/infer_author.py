"""
Strict batch author inference.

Input JSON schema (strict):
[
  { "id": "code1", "quote": "...", "true_author": "Virginia Woolf" },
  { "id": "code2", "quote": "...", "true_author": "Henry James" }
]

Usage:
python -m scripts.infer_author \
  --ckpt models/best.pt \
  --refs data/author_refs.json \
  --inputs data/batch_quotes.json \
  --out results/batch_results.json \
  [--summary]
"""

import argparse, json
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from src.model import SiameseModel


# refs to centroids
@torch.inference_mode()
def _load_centroids(ref_path: Path, model: SiameseModel, device, batch_size: int = 32):
    raw = ref_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "author" in data and "text" in data:
            items = [data]
        elif isinstance(data, dict):
            items = [{"author": a, "text": t} for a, texts in data.items() for t in texts]
        else:
            raise ValueError("Unsupported refs JSON structure.")
    except json.JSONDecodeError:
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]

    buckets = defaultdict(list)
    for it in items:
        buckets[str(it["author"])].append(it["text"])

    centroids = {}
    for author, texts in buckets.items():
        embs = []
        for i in range(0, len(texts), batch_size):
            e = model.encode(texts[i:i+batch_size]).to(device)  # [B, D]
            e = F.normalize(e, p=2, dim=-1)
            embs.append(e)
        E = torch.cat(embs, dim=0)
        c = F.normalize(E.mean(0), p=2, dim=-1)
        centroids[author] = c
    return centroids


# batch ranking 
@torch.inference_mode()
def _rank_authors_batch(
    texts: List[str],
    centroids: dict,
    model: SiameseModel,
    device,
    top_k: int = 3,
    batch_size: int = 32,
):
    X = model.encode(texts, batch_size=batch_size).to(device)  # [N, D]
    X = F.normalize(X, p=2, dim=-1)

    names, C = zip(*centroids.items())  # tuple[str], tuple[tensor]
    C = torch.stack(C)                  # [A, D]
    S = X @ C.T                         # cosine since unit-norm

    k = min(top_k, C.size(0))
    conf, idx = torch.topk(S, k=k, dim=1, largest=True, sorted=True)  # [N, k]
    conf = conf.cpu().numpy()
    idx = idx.cpu().numpy()

    top1 = [names[i[0]] for i in idx]
    top1_conf = [float(c[0]) for c in conf]
    topk_full = [
        [{"author": names[j], "score": float(conf[i, t])} for t, j in enumerate(idx[i])]
        for i in range(len(texts))
    ]
    return top1, top1_conf, topk_full


# IO helpers 
def _read_strict_inputs(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array of objects.")
    ids, quotes, golds = [], [], []
    for i, it in enumerate(data):
        if not isinstance(it, dict):
            raise ValueError(f"Item {i} is not an object.")
        if not all(k in it for k in ("id", "quote", "true_author")):
            missing = [k for k in ("id", "quote", "true_author") if k not in it]
            raise ValueError(f"Item {i} missing keys: {missing}")
        ids.append(str(it["id"]))
        quotes.append(str(it["quote"]))
        golds.append(str(it["true_author"]))
    return ids, quotes, golds


#  CLI 
def _cli():
    p = argparse.ArgumentParser(description="Strict batch author inference.")
    p.add_argument("--ckpt", required=True, help="Path to trained checkpoint (best.pt)")
    p.add_argument("--refs", required=True, help="author_refs.json (or JSONL)")
    p.add_argument("--inputs", required=True, help="Strict JSON with fields: id, quote, true_author")
    p.add_argument("--top-k", type=int, default=3)              # keep at 3 for your use
    p.add_argument("--summary", action="store_true",            # optional accuracy
                help="Include overall accuracy if true_author is provided.")

    p.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    p.add_argument("--out", required=True, help="Where to save JSON results")
    args = p.parse_args()

    # device
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    # load checkpoint (PyTorch 2.6-safe)
    try:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(args.ckpt, map_location="cpu")
    raw_weights = state.get("model_state_dict", state)
    weights = {k.replace("module.", "", 1): v for k, v in raw_weights.items()}

    # build model
    model = SiameseModel(
        encoder_name="all-MiniLM-L6-v2",
        proj_dim=256, mlp_hidden=512, dropout=0.1, init_temp=10.0, device=device
    ).to(device)
    model.load_state_dict(weights, strict=False)
    model.eval()

    # centroids
    centroids = _load_centroids(Path(args.refs), model, device, batch_size=args.batch_size)

    # inputs (strict)
    ids, quotes, gold = _read_strict_inputs(Path(args.inputs))

    # infer
    top1, top1_conf, topk = _rank_authors_batch(
        quotes, centroids, model, device, top_k=3, batch_size=args.batch_size  # force top-3
    )

    # minimal results: id, pred, top3 (no quote text)
    results = []
    correct = 0
    for i in range(len(quotes)):
        results.append({
            "id": ids[i],
            "pred": top1[i],
            "top3": topk[i][:3],  # [{author, score}, ...] length 3
        })
        if args.summary:
           # true_author always present
            correct += int(gold[i] == top1[i])

    # optional summary
    if args.summary:
        acc = correct / max(1, len(quotes))
        payload = {"results": results, "summary": {"count": len(quotes), "accuracy": acc}}
    else:
        payload = results

    # write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # console print
    print(f"Wrote {len(results)} rows to {out_path}")
    if args.summary:
        print(f"Accuracy: {acc:.3f}")

if __name__ == "__main__":
    _cli()
