
"""
    This file defines TF‑IDF + Logistic Regression utilities.

Exposes:
- preprocess_batch(texts)
- fit_tfidf_logreg(X_train, y_train, *, seed=42, tfidf_kwargs=None, C=1.0)
- predict(vectorizer, clf, X_texts)
- encode(vectorizer, X_texts, svd_dim=128, seed=42) -> dense embeddings for t‑SNE
- save_artifacts(out_dir, vectorizer, clf)
- load_artifacts(out_dir) -> (vectorizer, clf)
"""

import os
from typing import List, Tuple, Dict, Any, Optional
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    _ = word_tokenize("ok")
except LookupError:
    nltk.download("punkt")
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# lowercase, keep alphabetic tokens, remove stopwords, Porter stem, join with spaces
def preprocess_batch(texts):
    sw = set(stopwords.words("english"))
    stem = PorterStemmer()

    def proc(t):
        words = []
        for s in sent_tokenize(t.strip()):
            toks = word_tokenize(s)
            toks = [w.lower() for w in toks if w.isalpha()]
            toks = [w for w in toks if w not in sw]
            toks = [stem.stem(w) for w in toks]
            words.extend(toks)
        return " ".join(words)

    return [proc(t) for t in texts]

# fit TF‑IDF and logistic regression
def fit_tfidf_logreg(X_train_texts, y_train, seed=42, tfidf_kwargs= None, C=1.0):
    tfidf_kwargs = tfidf_kwargs or {}
    vec = TfidfVectorizer(**tfidf_kwargs)
    Xtr = vec.fit_transform(X_train_texts)

    clf = LogisticRegression(max_iter=10000, random_state=seed, C=C)
    clf.fit(Xtr, y_train)
    return vec, clf

# perdict
def predict(vec, clf, X_texts):
    X = vec.transform(X_texts)
    return clf.predict(X).tolist()

# return dense projections of TF‑IDF features for plotting
def encode(vec, X_texts, svd_dim=128, seed= 42):
    X = vec.transform(X_texts)
    k = max(2, min(svd_dim, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=k, random_state=seed)
    Z = svd.fit_transform(X)
    return Z

def save_artifacts(out_dir: str, vec: TfidfVectorizer, clf: LogisticRegression) -> None:
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(vec, os.path.join(out_dir, "vectorizer.joblib"))
    joblib.dump(clf, os.path.join(out_dir, "classifier.joblib"))

def load_artifacts(out_dir: str) -> Tuple[TfidfVectorizer, LogisticRegression]:
    vec = joblib.load(os.path.join(out_dir, "vectorizer.joblib"))
    clf = joblib.load(os.path.join(out_dir, "classifier.joblib"))
    return vec, clf
