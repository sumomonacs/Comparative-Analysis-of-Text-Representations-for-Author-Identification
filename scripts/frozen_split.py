'''
    This file freeze a train/test split for consistent evaluation across all models.
    Return a json file with testing index.
'''

import os
import sys
import json
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.model_selection import train_test_split
from scripts.config import EXCERPT_FILE, TRAIN_DATA

SEED = 42
TEST_SIZE = 0.2

def main():
    os.makedirs(os.path.dirname(TRAIN_DATA), exist_ok=True)

    df = pd.read_csv(EXCERPT_FILE)
    assert "author" in df.columns, "CSV must have an 'author' column for stratified split."

    # stratified split by author so all appear in both sets
    _, test_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["author"]
    )

    split = {"seed": SEED, "test_index": sorted(int(i) for i in test_idx)}

    with open(TRAIN_DATA, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    print(f"Frozen split saved to {TRAIN_DATA}")
    print(f"Test set size: {len(split['test_index'])} / {len(df)} total rows")

if __name__ == "__main__":
    main()