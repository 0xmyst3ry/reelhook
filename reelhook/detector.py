from dataclasses import dataclass, field

from .data import generate
from .htmlscan import analyze
from .model import LogisticRegression
from .urlfeatures import FEATURE_NAMES, vector

_DEFAULT = None


def default_model() -> LogisticRegression:
    """Train on the bundled synthetic corpus (deterministic, ~1s)."""
    global _DEFAULT
    if _DEFAULT is None:
        rows = generate(3000, seed=1)
        X = [vector(u) for u, _ in rows]
        y = [l for _, l in rows]
        _DEFAULT = LogisticRegression(FEATURE_NAMES).fit(X, y, epochs=60)
    return _DEFAULT


@dataclass
class Report:
    url: str
    score: float
    verdict: str  # safe | suspicious | phishing
    url_score: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self):
        return {"url": self.url, "score": round(self.score, 3), "verdict": self.verdict,
                "url_score": round(self.url_score, 3), "reasons": self.reasons}


def check(url: str, html: str | None = None, model: LogisticRegression | None = None) -> Report:
    model = model or default_model()
    x = vector(url)
    p_url = model.predict_proba(x)
    reasons = [f"{n} ({c:+.2f})" for n, c in model.explain(x) if abs(c) > 0.15 and c > 0]

    miss = 1 - p_url
    if html is not None:
        for s in analyze(html, url):
            miss *= 1 - s.weight
            reasons.append(f"[{s.id}] {s.note}")
    score = 1 - miss
    verdict = "phishing" if score >= 0.75 else "suspicious" if score >= 0.45 else "safe"
    return Report(url, score, verdict, p_url, reasons)
