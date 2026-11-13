"""Hashed character n-gram logistic regression on the raw URL string.

Learns lexical patterns no hand-written feature anticipates (kit paths, odd token shapes).
Sparse SGD, so the model is a small JSON dict of non-zero weights.
"""
import math
import random
import re
import zlib

from .urlfeatures import decode_idna, parse, skeleton

DIM = 1 << 17
_SPLIT = re.compile(r"[^a-z0-9]+")


def _h(s: str) -> int:
    return zlib.crc32(s.encode()) & (DIM - 1)


def featurize(url: str) -> dict:
    u = parse(url)
    host = decode_idna((u.hostname or "").lower())
    text = f"{host}|{(u.path or '').lower()}|{(u.query or '').lower()}"[:220]
    counts: dict = {}

    def add(key):
        i = _h(key)
        counts[i] = counts.get(i, 0.0) + 1.0

    padded = f"^{text}$"
    for n in (3, 4, 5):
        for i in range(len(padded) - n + 1):
            add(f"c{n}:{padded[i:i + n]}")
    for tok in _SPLIT.findall(text):
        add("t:" + tok)
        sk = skeleton(tok)
        if sk != tok:
            add("s:" + sk)
    vals = {k: 1 + math.log(v) for k, v in counts.items()}
    norm = math.sqrt(sum(v * v for v in vals.values())) or 1.0
    return {k: v / norm for k, v in vals.items()}


def _sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


class NgramModel:
    def __init__(self):
        self.w, self.b = {}, 0.0

    def decision(self, x):
        return sum(self.w.get(i, 0.0) * v for i, v in x.items()) + self.b

    def predict_proba(self, url: str) -> float:
        return _sigmoid(self.decision(featurize(url)))

    def fit(self, rows, epochs=10, lr=0.6, l2=2e-5, seed=3):
        data = [(featurize(u), y) for u, y in rows]
        rng = random.Random(seed)
        for ep in range(epochs):
            rng.shuffle(data)
            step = lr / (1 + 0.5 * ep)
            for x, y in data:
                err = _sigmoid(self.decision(x)) - y
                for i, v in x.items():
                    w = self.w.get(i, 0.0)
                    self.w[i] = w - step * (err * v + l2 * w)
                self.b -= step * err
        self.w = {i: w for i, w in self.w.items() if abs(w) > 1e-3}
        return self

    def to_dict(self):
        return {"b": self.b, "w": {str(i): round(w, 3) for i, w in self.w.items()}}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.b, m.w = d["b"], {int(i): w for i, w in d["w"].items()}
        return m
