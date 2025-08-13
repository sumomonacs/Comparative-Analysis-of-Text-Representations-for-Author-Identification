"""
    This file defines stylometric feature extractor.
"""

import math
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# ensure required NLTK resources are available
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# punctuation set
_PUNCT = list(".,;:!?\"'()[]{}-—…“”‘’")

# safe division in case the denominator is 0
def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0

#  compute TTR, hapax ratio, Honoré's R, and Yule's K
def _lexical_measures(tokens_lower):
    N = len(tokens_lower)
    freqs = Counter(tokens_lower)
    V = len(freqs)
    V1 = sum(1 for _, c in freqs.items() if c == 1)

    ttr = _safe_div(V, N)
    hapax_ratio = _safe_div(V1, V)

    honore_R = 0.0
    if V and V1 < V and N > 0:
        honore_R = 100.0 * math.log(max(N, 1)) / (1.0 - (V1 / V))

    if N > 0:
        v_of_i = Counter(freqs.values())
        sum_i2vi = sum((i * i) * vi for i, vi in v_of_i.items())
        yule_K = 1e4 * _safe_div((sum_i2vi - N), (N * N))
    else:
        yule_K = 0.0

    return ttr, hapax_ratio, honore_R, yule_K


class StylometricVectorizer(BaseEstimator, TransformerMixin):
    """
    handcrafted stylometric features:
      - sentence stats (avg len in tokens/chars)
      - token stats (avg token len, stopword %)
      - lexical richness (TTR, hapax ratio, Honoré's R, Yule's K)
      - punctuation normalized counts
      - function-word normalized frequencies 
    """
    def __init__(
        self,
        function_words=None,
        lowercase=True,
        keep_alpha_only=True,
        include_punct=True,
        include_function_word_vector=True,
        max_function_words=150,
        random_state= 42,
    ):
        self.function_words = list(function_words) if function_words else None
        self.lowercase = lowercase
        self.keep_alpha_only = keep_alpha_only
        self.include_punct = include_punct
        self.include_function_word_vector = include_function_word_vector
        self.max_function_words = max_function_words
        self.random_state = random_state

        self._fword_vocab_ = []
        self.feature_names_ = []
        self._sw_set = set(stopwords.words("english"))

    def fit(self, texts, y=None):
        if self.include_function_word_vector:
            if self.function_words is None:
                fw = self._sw_set | {
                    "’s","’d","’re","’ll","’ve","’m",
                    "could","would","should","shall","might","must","ought",
                    "do","does","did","doing","done",
                    "be","is","am","are","was","were","being","been",
                    "have","has","had","having",
                    "will","can","may",
                    "not","no","nor",
                    "there","here","thus","hence","where","when","while","whilst",
                    "upon","unto","into","onto","within","without",
                    "also","even","ever","yet","rather","quite","almost",
                }
            else:
                fw = set(w.lower() for w in self.function_words)

            counts = Counter()
            for t in texts:
                toks = self._tokenize(t)
                toks_fw = [w for w in toks if w in fw]
                counts.update(toks_fw)

            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if not ranked:
                ranked = [(w, 1) for w in sorted(fw)]
            self._fword_vocab_ = [w for w, _ in ranked[: self.max_function_words]]
        else:
            self._fword_vocab_ = []

        self.feature_names_ = self._build_feature_names()
        return self

    def transform(self, texts):
        feats = [self._featurize_one(t) for t in texts]
        return np.vstack(feats).astype(np.float32)

    def _build_feature_names(self):
        base = [
            "num_chars",
            "num_tokens",
            "num_types",
            "avg_token_len",
            "avg_sent_len_tokens",
            "avg_sent_len_chars",
            "stopword_ratio",
            "ttr",
            "hapax_ratio",
            "honore_R",
            "yule_K",
        ]
        punct_feats = [f"punct_freq[{p}]" for p in _PUNCT] if self.include_punct else []
        fword_feats = [f"fw[{w}]" for w in self._fword_vocab_] if self.include_function_word_vector else []
        return base + punct_feats + fword_feats

    def _tokenize(self, text):
        toks = word_tokenize(text)
        if self.lowercase:
            toks = [t.lower() for t in toks]
        if self.keep_alpha_only:
            toks = [t for t in toks if t.isalpha()]
        return toks

    def _featurize_one(self, text):
        sents = sent_tokenize(text) or [text]
        sent_token_counts, sent_char_counts = [], []
        for s in sents:
            toks_s = word_tokenize(s)
            if self.lowercase:
                toks_s = [t.lower() for t in toks_s]
            toks_alpha = [t for t in toks_s if (t.isalpha() if self.keep_alpha_only else True)]
            sent_token_counts.append(len(toks_alpha))
            sent_char_counts.append(len(s))

        tokens_lower = self._tokenize(text)
        num_tokens = len(tokens_lower)
        num_chars = len(text)
        num_types = len(set(tokens_lower))
        avg_token_len = _safe_div(sum(len(t) for t in tokens_lower), num_tokens)
        avg_sent_len_tokens = _safe_div(sum(sent_token_counts), len(sents))
        avg_sent_len_chars = _safe_div(sum(sent_char_counts), len(sents))
        stopword_ratio = _safe_div(sum(1 for t in tokens_lower if t in self._sw_set), num_tokens)

        ttr, hapax_ratio, honore_R, yule_K = _lexical_measures(tokens_lower)

        features = [
            float(num_chars),
            float(num_tokens),
            float(num_types),
            float(avg_token_len),
            float(avg_sent_len_tokens),
            float(avg_sent_len_chars),
            float(stopword_ratio),
            float(ttr),
            float(hapax_ratio),
            float(honore_R),
            float(yule_K),
        ]

        if self.include_punct:
            char_counts = Counter(ch for ch in text)
            denom = max(num_chars, 1)
            for p in _PUNCT:
                features.append(_safe_div(char_counts.get(p, 0), denom))

        if self.include_function_word_vector and self._fword_vocab_:
            freqs = Counter(tokens_lower)
            denom = max(num_tokens, 1)
            for w in self._fword_vocab_:
                features.append(_safe_div(freqs.get(w, 0), denom))

        return np.array(features, dtype=np.float32)
