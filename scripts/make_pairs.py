import os
import json
import sys
import random
import pandas as pd
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.config import PAIR_FILE, EXCERPT_FILE

# build a contrastive-learning pair list with author metadata.
def generate_contrastive_pairs(df, pos_per_author=1000, neg_pairs=5000, seed=42):
    random.seed(seed)

    # map author to list
    label_to_texts = {author: df[df.author == author].excerpt.tolist() for author in df.author.unique()}
    authors = list(label_to_texts)

    pairs = []

    #positive pairs (same author) 
    for author, texts in label_to_texts.items():
        # upper-bound by combinatorial maximum
        needed = min(pos_per_author, len(texts) * (len(texts) - 1) // 2)
        for _ in range(needed):
            a, b = random.sample(texts, 2)
            pairs.append((a, b, 1, author, author))

    # negative pairs (different authors)
    for _ in range(neg_pairs):
        a1, a2 = random.sample(authors, 2)     # uniform over author pairs
        pairs.append((random.choice(label_to_texts[a1]), random.choice(label_to_texts[a2]), 0, a1, a2))

    random.shuffle(pairs)
    return pairs

def main():
    df = pd.read_csv(EXCERPT_FILE)

    # 1000 positives per author  
    pairs = generate_contrastive_pairs(df, pos_per_author=1000, neg_pairs=5000,seed=42)

    # save as a JSON list of dicts
    out_path = Path(PAIR_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            [{"text1": a, "text2": b, "label": lbl, "author1": author1, "author2": author2} for a, b, lbl, author1, author2 in pairs],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # quick summary
    n_pos = sum(lbl for _, _, lbl, _, _ in pairs)
    n_neg = len(pairs) - n_pos
    print(f"Saved {len(pairs):,} pairs to {out_path}  (pos: {n_pos:,}  neg: {n_neg:,})")

if __name__ == "__main__":
    main()