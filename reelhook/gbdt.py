"""Gradient-boosted decision trees for binary classification, from scratch.

Newton boosting on the logistic loss (as in XGBoost): each tree is fit to the gradient g and
hessian h of the loss, leaf value = -G / (H + lambda), split gain from the second-order
objective. Features are pre-binned into histograms so growing a tree is O(rows x features)
per level. Predictions can be explained per feature with Saabas path attributions.
"""
import bisect
import math
import random


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


class GBDT:
    def __init__(self, names, n_trees=80, depth=3, lr=0.15, lam=1.0, min_leaf=8, bins=24, subsample=0.8, seed=1):
        self.names, self.n_trees, self.depth, self.lr = names, n_trees, depth, lr
        self.lam, self.min_leaf, self.bins, self.subsample, self.seed = lam, min_leaf, bins, subsample, seed
        self.trees, self.base = [], 0.0

    # -- training ------------------------------------------------------------
    def _edges(self, col):
        vals = sorted(set(col))
        if len(vals) <= self.bins:
            return [(a + b) / 2 for a, b in zip(vals, vals[1:])]
        s = sorted(col)
        cuts = sorted({s[int(len(s) * q / self.bins)] for q in range(1, self.bins)})
        return [c for c in cuts if c > s[0]]

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        rng = random.Random(self.seed)
        self.edges = [self._edges([row[f] for row in X]) for f in range(d)]
        cols = [[bisect.bisect_left(self.edges[f], row[f]) for row in X] for f in range(d)]
        pos = sum(y)
        self.base = math.log((pos + 1) / (n - pos + 1))
        F = [self.base] * n
        self.trees = []
        for _ in range(self.n_trees):
            p = [_sigmoid(f) for f in F]
            g = [pi - yi for pi, yi in zip(p, y)]
            h = [max(pi * (1 - pi), 1e-6) for pi in p]
            rows = [i for i in range(n) if rng.random() < self.subsample] or list(range(n))
            tree = self._grow(rows, self.depth, g, h, cols)
            self.trees.append(tree)
            for i in range(n):
                F[i] += self.lr * self._leaf(tree, X[i])
        return self

    def _grow(self, rows, depth, g, h, cols):
        G = sum(g[i] for i in rows)
        H = sum(h[i] for i in rows)
        value = -G / (H + self.lam)
        if depth == 0 or len(rows) < 2 * self.min_leaf:
            return [value]
        parent = G * G / (H + self.lam)
        best = (1e-6, None, None)
        for f, col in enumerate(cols):
            nb = len(self.edges[f]) + 1
            if nb < 2:
                continue
            hg, hh, hc = [0.0] * nb, [0.0] * nb, [0] * nb
            for i in rows:
                b = col[i]
                hg[b] += g[i]
                hh[b] += h[i]
                hc[b] += 1
            gl = hl = 0.0
            cl = 0
            for b in range(nb - 1):
                gl += hg[b]
                hl += hh[b]
                cl += hc[b]
                if cl < self.min_leaf or len(rows) - cl < self.min_leaf:
                    continue
                gain = gl * gl / (hl + self.lam) + (G - gl) ** 2 / (H - hl + self.lam) - parent
                if gain > best[0]:
                    best = (gain, f, b)
        if best[1] is None:
            return [value]
        _, f, b = best
        left = [i for i in rows if cols[f][i] <= b]
        right = [i for i in rows if cols[f][i] > b]
        return [f, self.edges[f][b], value,
                self._grow(left, depth - 1, g, h, cols), self._grow(right, depth - 1, g, h, cols)]

    # -- inference -----------------------------------------------------------
    @staticmethod
    def _leaf(tree, x):
        while len(tree) > 1:
            tree = tree[3] if x[tree[0]] <= tree[1] else tree[4]
        return tree[0]

    def decision(self, x) -> float:
        return self.base + self.lr * sum(self._leaf(t, x) for t in self.trees)

    def predict_proba(self, x) -> float:
        return _sigmoid(self.decision(x))

    def explain(self, x, top=5):
        """Saabas attributions: log-odds contribution of each feature along the paths taken."""
        contrib = [0.0] * len(self.names)
        for tree in self.trees:
            node = tree
            while len(node) > 1:
                child = node[3] if x[node[0]] <= node[1] else node[4]
                child_value = child[2] if len(child) > 1 else child[0]
                contrib[node[0]] += self.lr * (child_value - node[2])
                node = child
        ranked = sorted(zip(self.names, contrib), key=lambda t: -abs(t[1]))
        return [(n, c) for n, c in ranked[:top] if abs(c) > 1e-9]

    # -- persistence -----------------------------------------------------------
    def to_dict(self):
        return {"names": self.names, "base": self.base, "lr": self.lr, "trees": self.trees}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["names"], lr=d["lr"])
        m.base, m.trees = d["base"], d["trees"]
        return m
