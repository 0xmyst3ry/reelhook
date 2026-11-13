"""Head-to-head model comparison, including a distribution-shift test on hand-written URLs."""
import os

from .data import generate, load_csv
from .ensemble import Ensemble
from .gbdt import GBDT
from .model import LogisticRegression, auc, metrics, tpr_at_fpr
from .ngram import NgramModel
from .urlfeatures import FEATURE_NAMES, vector

HOLDOUT = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "urls_holdout.csv")


def _score_table(name, rows, scorers, out):
    y = [l for _, l in rows]
    out(f"\n{name}: {sum(y)} phishing / {len(y) - sum(y)} legitimate")
    out(f"{'model':<18}{'AUC':>7}{'TPR@1%FPR':>11}{'precision':>11}{'recall':>8}{'F1':>7}")
    res = {}
    for label, fn in scorers.items():
        p = [fn(u) for u, _ in rows]
        m = metrics(y, p)
        res[label] = {**m, "tpr1": tpr_at_fpr(y, p, 0.01)}
        out(f"{label:<18}{m['auc']:>7.3f}{res[label]['tpr1']:>11.3f}{m['precision']:>11.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}")
    return res


def cross_validate(rows, k=5, n_trees=40):
    """k-fold AUC for the feature-based models on the synthetic corpus."""
    folds = [rows[i::k] for i in range(k)]
    out = {"lr": [], "gbdt": []}
    for i in range(k):
        test = folds[i]
        train = [r for j, f in enumerate(folds) if j != i for r in f]
        X, y = [vector(u) for u, _ in train], [l for _, l in train]
        lr = LogisticRegression(FEATURE_NAMES).fit(X, y, epochs=30)
        gb = GBDT(FEATURE_NAMES, n_trees=n_trees).fit(X, y)
        yt = [l for _, l in test]
        out["lr"].append(auc(yt, [lr.predict_proba(vector(u)) for u, _ in test]))
        out["gbdt"].append(auc(yt, [gb.predict_proba(vector(u)) for u, _ in test]))
    return {k_: (sum(v) / len(v), min(v), max(v)) for k_, v in out.items()}


def run(train_csv=None, test_csv=None, out=print, ensemble: Ensemble | None = None):
    train = load_csv(train_csv) if train_csv else generate(3000, seed=1)
    valid = generate(1000, seed=2)
    synth = load_csv(test_csv) if test_csv else generate(1000, seed=99)
    holdout = load_csv(HOLDOUT)
    ens = ensemble or Ensemble.train(train, valid)
    scorers = {
        "logistic (feat)": lambda u: ens.lr.predict_proba(vector(u)),
        "GBDT (feat)": lambda u: ens.gbdt.predict_proba(vector(u)),
        "char n-gram": lambda u: ens.ngram.predict_proba(u),
        "ensemble": ens.predict_proba,
    }
    r1 = _score_table("A. held-out synthetic data (same generator as training)", synth, scorers, out)
    r2 = _score_table("B. hand-written realistic URLs (different author, different style)", holdout, scorers, out)
    cv = cross_validate(train)
    out("\n5-fold CV AUC on training corpus: " + ", ".join(f"{k} {m:.3f} (min {lo:.3f})" for k, (m, lo, hi) in cv.items()))
    return ens, r1, r2
