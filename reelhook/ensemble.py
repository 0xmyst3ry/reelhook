"""Train, persist and combine the three URL models."""
import json
import os

from .data import generate
from .gbdt import GBDT
from .model import LogisticRegression, threshold_at_fpr
from .ngram import NgramModel
from .urlfeatures import FEATURE_NAMES, vector

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.json")


class Ensemble:
    def __init__(self, lr=None, gbdt=None, ngram=None, thresholds=None):
        self.lr, self.gbdt, self.ngram = lr, gbdt, ngram
        self.thresholds = thresholds or {"suspicious": 0.5, "phishing": 0.75}

    def probs(self, url: str) -> dict:
        x = vector(url)
        return {"lr": self.lr.predict_proba(x), "gbdt": self.gbdt.predict_proba(x), "ngram": self.ngram.predict_proba(url)}

    def predict_proba(self, url: str) -> float:
        p = self.probs(url)
        return (p["gbdt"] + p["ngram"] + p["lr"]) / 3

    def explain(self, url: str, top=4):
        return [(n, c) for n, c in self.gbdt.explain(vector(url), top) if c > 0.15]

    @classmethod
    def train(cls, train_rows=None, valid_rows=None, n_trees=80, seed=1):
        train_rows = train_rows or generate(3000, seed=seed)
        valid_rows = valid_rows or generate(1000, seed=seed + 1)
        X, y = [vector(u) for u, _ in train_rows], [l for _, l in train_rows]
        lr = LogisticRegression(FEATURE_NAMES).fit(X, y, epochs=40)
        gb = GBDT(FEATURE_NAMES, n_trees=n_trees, seed=seed).fit(X, y)
        ng = NgramModel().fit(train_rows)
        ens = cls(lr, gb, ng)
        yv = [l for _, l in valid_rows]
        pv = [ens.predict_proba(u) for u, _ in valid_rows]
        # alert thresholds come from held-out data: 5% and 1% false-positive rate, clamped to sane bounds
        ens.thresholds = {"suspicious": min(max(threshold_at_fpr(yv, pv, 0.05), 0.35), 0.6),
                          "phishing": min(max(threshold_at_fpr(yv, pv, 0.01), 0.6), 0.9)}
        return ens

    def save(self, path=MODEL_PATH):
        lr = self.lr
        with open(path, "w") as f:
            json.dump({"lr": {"names": lr.names, "w": lr.w, "b": lr.b, "mu": lr.mu, "sd": lr.sd},
                       "gbdt": self.gbdt.to_dict(), "ngram": self.ngram.to_dict(), "thresholds": self.thresholds}, f)

    @classmethod
    def load(cls, path=MODEL_PATH):
        with open(path) as f:
            d = json.load(f)
        lr = LogisticRegression(d["lr"]["names"])
        lr.__dict__.update(d["lr"])
        return cls(lr, GBDT.from_dict(d["gbdt"]), NgramModel.from_dict(d["ngram"]), d["thresholds"])


_DEFAULT = None


def default_model() -> Ensemble:
    """Shipped weights if present; otherwise train on the bundled generator (slower)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Ensemble.load() if os.path.exists(MODEL_PATH) else Ensemble.train()
    return _DEFAULT
