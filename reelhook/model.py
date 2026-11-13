"""Logistic regression with standardisation and L2, implemented from scratch."""
import json
import math
import random


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


class LogisticRegression:
    def __init__(self, names: list[str]):
        self.names = names
        self.w = [0.0] * len(names)
        self.b = 0.0
        self.mu = [0.0] * len(names)
        self.sd = [1.0] * len(names)

    def _z(self, x):
        return [(v - m) / s for v, m, s in zip(x, self.mu, self.sd)]

    def fit(self, X, y, epochs=300, lr=0.1, l2=1e-3, batch=64, seed=7):
        n, d = len(X), len(X[0])
        self.mu = [sum(r[j] for r in X) / n for j in range(d)]
        self.sd = [max(math.sqrt(sum((r[j] - self.mu[j]) ** 2 for r in X) / n), 1e-6) for j in range(d)]
        Z = [self._z(r) for r in X]
        rng = random.Random(seed)
        idx = list(range(n))
        for _ in range(epochs):
            rng.shuffle(idx)
            for s in range(0, n, batch):
                chunk = idx[s:s + batch]
                gw = [0.0] * d
                gb = 0.0
                for i in chunk:
                    err = _sigmoid(sum(w * z for w, z in zip(self.w, Z[i])) + self.b) - y[i]
                    for j in range(d):
                        gw[j] += err * Z[i][j]
                    gb += err
                m = len(chunk)
                for j in range(d):
                    self.w[j] -= lr * (gw[j] / m + l2 * self.w[j])
                self.b -= lr * gb / m
        return self

    def predict_proba(self, x) -> float:
        z = self._z(x)
        return _sigmoid(sum(w * v for w, v in zip(self.w, z)) + self.b)

    def explain(self, x, top=5):
        """Per-feature contribution to the logit, largest magnitude first."""
        z = self._z(x)
        contrib = [(n, w * v) for n, w, v in zip(self.names, self.w, z)]
        return sorted(contrib, key=lambda t: -abs(t[1]))[:top]

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        m = cls(d["names"])
        m.__dict__.update(d)
        return m


def metrics(y_true, p, threshold=0.5) -> dict:
    tp = sum(1 for t, q in zip(y_true, p) if t == 1 and q >= threshold)
    fp = sum(1 for t, q in zip(y_true, p) if t == 0 and q >= threshold)
    fn = sum(1 for t, q in zip(y_true, p) if t == 1 and q < threshold)
    tn = len(y_true) - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": (tp + tn) / len(y_true), "precision": prec, "recall": rec,
            "f1": f1, "auc": auc(y_true, p), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def auc(y_true, p) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U)."""
    pairs = sorted(zip(p, y_true))
    pos = sum(y_true)
    neg = len(y_true) - pos
    if not pos or not neg:
        return 0.0
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if pairs[k][1] == 1)
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def tpr_at_fpr(y_true, p, target_fpr=0.01) -> float:
    """Recall achievable while keeping the false-positive rate at or below `target_fpr`."""
    neg = sorted((q for t, q in zip(y_true, p) if t == 0), reverse=True)
    pos = [q for t, q in zip(y_true, p) if t == 1]
    if not neg or not pos:
        return 0.0
    allowed = int(target_fpr * len(neg))
    thr = neg[allowed] if allowed < len(neg) else 0.0   # scores strictly above thr stay within the FPR budget
    return sum(1 for q in pos if q > thr) / len(pos)


def threshold_at_fpr(y_true, p, target_fpr=0.01) -> float:
    neg = sorted((q for t, q in zip(y_true, p) if t == 0), reverse=True)
    allowed = int(target_fpr * len(neg))
    return neg[allowed] if allowed < len(neg) else 0.5
