"""
    This file dines the Word2Vec-based document embeddings + linear classifier

    - Trains a gensim Word2Vec model on my corpus
    - Builds document vectors by:
        * "mean"  : plain average of word vectors
        * "tfidf" : TF-IDF-weighted average (IDF learned from train set)
        * "sif"   : Smooth Inverse Frequency weighting + optional first PC removal

"""

import os
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.tokenize import word_tokenize
# ensure required tokenizer is available (runs once per environment)
for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

# --------------------------- helpers ---------------------------
def tfidf_tokenizer_default(s):
    return _tokenize(s, lowercase=True, alpha_only=True)

# tokenize
def _tokenize(text, lowercase=True, alpha_only= True):
    toks = word_tokenize(text)
    if lowercase:
        toks = [t.lower() for t in toks]
    if alpha_only:
        toks = [t for t in toks if t.isalpha()]
    return toks

# plain average of word vectors
def _doc_mean(vectors):
    if vectors.size == 0:
        return np.zeros((1, 0), dtype=np.float32)
    return vectors.mean(axis=0, keepdims=True)

#  TF-IDF-weighted average
def _weighted_mean(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return np.zeros((1, 0), dtype=np.float32)
    w = weights.reshape(-1, 1)
    wsum = float(np.sum(w))
    if wsum == 0:
        return _doc_mean(vectors)
    return (vectors * w).sum(axis=0, keepdims=True) / wsum

# word2vec configuration
@dataclass
class W2VConfig:
    vector_size: int = 300
    window: int = 5
    min_count: int = 3
    sg: int = 1                 # 0=CBOW, 1=skip-gram
    negative: int = 10
    epochs: int = 10
    workers: int = 4
    lowercase: bool = True
    alpha_only: bool = True
    doc_pool: str = "sif"       # "mean" | "tfidf" | "sif"
    sif_a: float = 1e-3         # SIF smoothing
    remove_first_pc: bool = True
    clf_C: float = 1.0
    seed: int = 42

class Word2VecDocEmbed:
    """
    rrain Word2Vec on a corpus and classify documents with a linear head.
    """

    def __init__(self, config= None):
        self.cfg = config or W2VConfig()
        self.w2v = None
        self.kv= None
        self.idf_vec= None
        self.pca= None

        # scale + classifie
        self.clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(max_iter=10000,
                               C=self.cfg.clf_C, random_state=self.cfg.seed),
        )
        self.dim_: int = self.cfg.vector_size
        self._fitted = False

    # ----------------------- public API -----------------------
    def fit(self, texts, labels):
        # train Word2Vec on tokenized corpus
        sents = [_tokenize(t, self.cfg.lowercase, self.cfg.alpha_only) for t in texts]
        self.w2v = Word2Vec(
            sentences=sents,
            vector_size=self.cfg.vector_size,
            window=self.cfg.window,
            min_count=self.cfg.min_count,
            sg=self.cfg.sg,
            negative=self.cfg.negative,
            epochs=self.cfg.epochs,
            workers=self.cfg.workers,
            seed=self.cfg.seed,
        )
        self.kv = self.w2v.wv
        self.dim_ = self.kv.vector_size

        # ptional IDF for tfidf/sif pooling
        if self.cfg.doc_pool in ("tfidf", "sif"):
            self.idf_vec = TfidfVectorizer(
                tokenizer=tfidf_tokenizer_default,
                lowercase=False, preprocessor=None, token_pattern=None, min_df=2
            )
            _ = self.idf_vec.fit(texts)

        # compute doc embeddings on train set
        X_emb = self._encode_docs(texts)

        # SIF first-PC removal
        if self.cfg.doc_pool == "sif" and self.cfg.remove_first_pc:
            self.pca = PCA(n_components=1, random_state=self.cfg.seed)
            self.pca.fit(X_emb)
            X_emb = self._remove_pc(X_emb, self.pca.components_[0])

        # fit classifier
        self.clf.fit(X_emb, labels)
        self._fitted = True
        return self

    def predict(self, texts):
        X = self.encode(texts)
        return self.clf.predict(X).tolist()

    def encode(self, texts):
        X_emb = self._encode_docs(texts)
        if self.cfg.doc_pool == "sif" and self.cfg.remove_first_pc and self.pca is not None:
            X_emb = self._remove_pc(X_emb, self.pca.components_[0])
        return X_emb

    def save(self, dirpath):
        os.makedirs(dirpath, exist_ok=True)
        # Word2Vec model
        if self.w2v is not None:
            self.w2v.save(os.path.join(dirpath, "w2v.model"))
        import joblib
        if self.idf_vec is not None:
            joblib.dump(self.idf_vec, os.path.join(dirpath, "idf_vectorizer.joblib"))
        if self.pca is not None:
            joblib.dump(self.pca, os.path.join(dirpath, "sif_pca.joblib"))
        # Classifier pipeline
        joblib.dump(self.clf, os.path.join(dirpath, "classifier.joblib"))

    @classmethod
    def load(cls, dirpath):
        import joblib
        obj = cls()
        obj.w2v = Word2Vec.load(os.path.join(dirpath, "w2v.model"))
        obj.kv = obj.w2v.wv
        obj.dim_ = obj.kv.vector_size
        idf_path = os.path.join(dirpath, "idf_vectorizer.joblib")
        pca_path = os.path.join(dirpath, "sif_pca.joblib")
        if os.path.exists(idf_path):
            obj.idf_vec = joblib.load(idf_path)
        if os.path.exists(pca_path):
            obj.pca = joblib.load(pca_path)
        obj.clf = joblib.load(os.path.join(dirpath, "classifier.joblib"))
        obj._fitted = True
        return obj

    # ----------------------- internals -----------------------
    def _word_vec(self, w):
        if self.kv is None:
            return None
        return self.kv[w] if w in self.kv else None

    def _encode_docs(self, texts):
        embs = []
        use_tfidf = (self.cfg.doc_pool == "tfidf") and (self.idf_vec is not None)
        use_sif = (self.cfg.doc_pool == "sif") and (self.idf_vec is not None)

        for text in texts:
            toks = _tokenize(text, self.cfg.lowercase, self.cfg.alpha_only)
            if not toks:
                embs.append(np.zeros(self.dim_, dtype=np.float32))
                continue

            vecs, weights = [], []
            for w in toks:
                v = self._word_vec(w)
                if v is None:
                    continue
                vecs.append(v)
                if use_tfidf:
                    weights.append(self._idf_weight(w))
                elif use_sif:
                    idf = self._idf_weight(w)
                    pw = 1.0 / (idf + 1e-9)  # proxy for p(w)
                    a = self.cfg.sif_a
                    weights.append(a / (a + pw))
                else:
                    weights.append(1.0)

            if not vecs:
                embs.append(np.zeros(self.dim_, dtype=np.float32))
                continue

            V = np.vstack(vecs)
            W = np.asarray(weights, dtype=np.float32)

            if self.cfg.doc_pool == "mean":
                emb = _doc_mean(V)
            else:
                emb = _weighted_mean(V, W)  # tfidf or sif
            embs.append(emb.squeeze(0))

        return np.vstack(embs).astype(np.float32)

    def _idf_weight(self, w):
        if self.idf_vec is None:
            return 1.0
        try:
            idx = self.idf_vec.vocabulary_.get(w, -1)
            if idx == -1:
                return float(np.median(self.idf_vec.idf_))
            return float(self.idf_vec.idf_[idx])
        except Exception:
            return 1.0

    @staticmethod
    def _remove_pc(X, pc):
        pc = pc.reshape(-1, 1)
        return X - (X @ pc) @ pc.T
