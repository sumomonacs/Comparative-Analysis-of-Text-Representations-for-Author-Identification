'''
    This file extracts excerpts from the books.
'''
import os
import re
import sys
import string
import random
import pandas as pd
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import CLEANED_DIR, EXCERPT_FILE, EVALUATE_CLEANED_DIR, INFERENCE_FILE

AUTHOR_MAP = {
    "bronte":  "Charlotte Brontë",
    "eliot":   "George Eliot",
    "james":   "Henry James",
    "woolf":   "Virginia Woolf",
    "wharton": "Edith Wharton",
}

# different criteria for shorter books(can extract across paragraphs)
BOOK_OVERRIDES = {
    "woolf_mrs_dalloway": {
        "extractor":  "multi_para",
        "min_words":  120,
        "max_paras":  6,
        "para_stride": 1,
    },
    "wharton_age_of_innocence": {
        "extractor":  "multi_para",
        "min_words":  180,
        "max_paras":  4,
        "para_stride": 1,
    },
    "james_turn_of_the_screw": {
        "extractor":  "multi_para",
        "min_words":  180,
        "max_paras":  5,
        "para_stride": 1,
    },
}

# get the excerpts inside the paragraph, and roughly uniform across the book
def split_into_excerpts(text, min_words=200, max_words=400, stride=100, buffer_from_end=500, max_excerpts=None, seed=42):
    random.seed(seed)

    # split into paragraphs and drop empties
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    # work out how many words total to skip the tail
    total_words = len(text.split())

    # keep only paragraphs that lie before the tail zone
    running_total = 0
    usable_paragraphs = []
    for p in paragraphs:
        running_total += len(p.split())
        if running_total > total_words - buffer_from_end:
            break
        usable_paragraphs.append(p)

    excerpts = []

    # sample within each usable paragraph
    for para in usable_paragraphs:
        words = para.split()
        if len(words) < min_words:
            continue

        starts = list(range(0, len(words) - min_words, stride))
        random.shuffle(starts)           # randomise start positions

        for start in starts:
            chunk = words[start : start + max_words]
            if len(chunk) >= min_words:
                excerpts.append(" ".join(chunk))
                break                    # take at most one per paragraph

    random.shuffle(excerpts)             # shuffle global order

    if max_excerpts is not None:
        excerpts = excerpts[:max_excerpts]

    return excerpts


# a helper function to define the heading form
_heading_re = re.compile(
    r"""^              # start
        (?:chapter|book|section)?  # optional label
        \s*                    # whitespace
        [ivxlcdm]+             # roman numerals
        [\.\s]*$               # optional dot/space, end
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

# determine standalone heading such as 'III' or 'CHAPTER XL'.
def _looks_like_heading(para):
    # strip punctuation & collapse spaces for robust matching
    clean = para.translate(str.maketrans("", "", string.punctuation)).strip()
    if _heading_re.match(clean):
        return True
    # ultra-short indicates a heading
    return len(clean.split()) <= 3


    
# concatenate up to max_para non-heading paragraphs to build excerpts, skip the final words to avoid spoilers
def multi_para_excerpts(text, min_words=200, max_words=400, max_paras=4,para_stride=1, buffer_from_end=20, max_excerpts=None, seed=42):
    random.seed(seed)

    # split paras & filter headings
    raw_paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    paras = [p for p in raw_paras if not _looks_like_heading(p)]

    total_words = len(text.split())
    spoiler_cutoff = total_words - buffer_from_end

    excerpts = []
    i, words_seen = 0, 0

    while i < len(paras) and words_seen < spoiler_cutoff:
        chunk, chunk_words, added = [], 0, 0
        for j in range(i, min(i + max_paras, len(paras))):
            p_words = paras[j].split()
            if words_seen + chunk_words + len(p_words) > spoiler_cutoff:
                break
            chunk.append(paras[j])
            chunk_words += len(p_words)
            added += 1
            if chunk_words >= min_words:
                break

        if min_words <= chunk_words <= max_words:
            excerpts.append(" ".join(chunk))

        # advance
        i += max(added, para_stride)
        words_seen += sum(len(p.split()) for p in paras[i - added : i])

        if max_excerpts and len(excerpts) >= max_excerpts:
            break

    random.shuffle(excerpts)
    return excerpts

# get author name from the file name
def author_from_fname(fname):
    for key, name in AUTHOR_MAP.items():
        if fname.startswith(key):
            return name
    return "Unknown"

# build the excerpt dataset
def build_excerpt_dataset(min_words=200, max_words=400, max_excerpts_per_book=120,
    max_excerpts_per_author=200, stride= 300):
    rng = random.Random(42)
    # collect files per author
    author_books = defaultdict(list)
    for fname in os.listdir(CLEANED_DIR):
        if fname.endswith(".txt"):
            author_books[author_from_fname(fname)].append(fname)

    rows = []

    # iterate author to book
    for author, books in author_books.items():
        per_book_cap = min(
            max_excerpts_per_book,
            max_excerpts_per_author // max(len(books), 1)
        )

        for fname in books:
            base = os.path.splitext(fname)[0]         
            path = os.path.join(CLEANED_DIR, fname)
            with open(path, encoding="utf-8") as f:
                text = f.read()

            override = BOOK_OVERRIDES.get(base)

            if override and override["extractor"] == "multi_para":
                excerpts = multi_para_excerpts(
                    text,
                    min_words   = override.get("min_words",   180),
                    max_words   = max_words,
                    max_paras   = override.get("max_paras",   4),
                    para_stride = override.get("para_stride", 1),
                    max_excerpts= per_book_cap,
                )
            else:
                excerpts = split_into_excerpts(
                    text,
                    min_words   = min_words,
                    max_words   = max_words,
                    stride      = stride,
                    max_excerpts= per_book_cap,
                )

            # append rows no matter which extractor was used
            for ex in excerpts:
                rows.append({"excerpt": ex, "author": author, "book": base})

    df = pd.DataFrame(rows)
    
    # sanity reports
    print("\nExcerpt distribution (by author)")
    print(df.groupby("author").size().to_string())

    print("\nExcerpt distribution (by book)")
    print(df.groupby(["author", "book"]).size().to_string())

    print("---------------------------------------")
    print(f"Total excerpts: {len(df)}\n")

    return df

# build the inference excerpt dataset
def build_inference_excerpts(buckets=None, stride=300, seed=7):
    
    if buckets is None:
        buckets = [
            {"label":"short", "min_words":50,  "max_words":100, "max_per_book":10, "max_per_author":20},
            {"label":"long",  "min_words":150, "max_words":300, "max_per_book":20, "max_per_author":40},
        ]

    rng = random.Random(seed)

    # collect files per author (sorted for determinism)
    author_books = defaultdict(list)
    for fname in sorted(os.listdir(EVALUATE_CLEANED_DIR)):
        if fname.endswith(".txt"):
            author_books[author_from_fname(fname)].append(fname)

    rows = []
    global_seen = set() 

    for bucket in buckets:
        label         = bucket["label"]
        min_words     = bucket["min_words"]
        max_words     = bucket["max_words"]
        max_per_book  = bucket["max_per_book"]
        max_per_author= bucket["max_per_author"]

        for author, books in author_books.items():
            per_book_cap = min(
                max_per_book,
                max_per_author // max(len(books), 1)
            )

            collected_this_author = 0

            for fname in sorted(books):
                if collected_this_author >= max_per_author:
                    break

                base = os.path.splitext(fname)[0]
                path = os.path.join(EVALUATE_CLEANED_DIR, fname)
                with open(path, encoding="utf-8") as f:
                    text = f.read()

                # gather many, then cap after shuffle
                excerpts = split_into_excerpts(
                    text,
                    min_words=min_words,
                    max_words=max_words,
                    stride=stride,
                    max_excerpts=None
                )

                rng.shuffle(excerpts)
                take = min(per_book_cap, max_per_author - collected_this_author)

                picked = []
                for ex in excerpts:
                    if len(picked) >= take:
                        break
                    key = (base, " ".join(ex.split()).lower())  # normalize for de-dup across buckets
                    if key in global_seen:
                        continue
                    global_seen.add(key)
                    picked.append(ex)

                for ex in picked:
                    rows.append({"excerpt": ex, "author": author, "book": base, "bucket": label})

                collected_this_author += len(picked)

    df = pd.DataFrame(rows)

    # sanity reports
    print("\nExcerpt distribution (by author)")
    print(df.groupby("author").size().to_string())

    print("\nExcerpt distribution (by book)")
    print(df.groupby(["author", "book"]).size().to_string())

    print("\nExcerpt distribution (by length)")
    print(df.groupby("bucket").size().to_string())

    print("---------------------------------------")
    print(f"Total excerpts: {len(df)}\n")

    return df

def main():
    df = build_excerpt_dataset()
    df.to_csv(EXCERPT_FILE, index=False)
    print(f"Saved to {EXCERPT_FILE}")

    df_infer = build_inference_excerpts(
        buckets=[
            {"label":"short", "min_words":50,  "max_words":100, "max_per_book":10, "max_per_author":20},
            {"label":"long",  "min_words":150, "max_words":300, "max_per_book":20, "max_per_author":40},
        ],
        stride=300,
        seed=7
    )
    df_infer.to_csv(INFERENCE_FILE, index=False)
    print(f"Saved to {INFERENCE_FILE}")

if __name__ == "__main__":
    main()
