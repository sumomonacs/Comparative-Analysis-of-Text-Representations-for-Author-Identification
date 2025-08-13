'''
    This file builds author reference dataset for later inference.
'''
import pandas as pd
import json
from pathlib import Path


def main():
    # configuration
    csv_path = Path("data/excerpts.csv")          
    out_json = Path("data/author_refs.json")      
    excerpts_per_author = 50                      

    # load data
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["author", "excerpt"])

    # sample and collect references
    refs = []
    for author, group in df.groupby("author"):
        sampled = group.sample(
            n=min(excerpts_per_author, len(group)),
            random_state=42
        )
        for _, row in sampled.iterrows():
            refs.append({
                "author": author,
                "text": row["excerpt"]
            })

    # write to JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(refs, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(refs)} references to {out_json}")

if __name__ == "__main__":
    main()
