from dataclasses import dataclass, field

from .emailscan import EmailReport, analyze_email
from .ensemble import Ensemble, default_model
from .htmlscan import analyze


@dataclass
class Report:
    url: str
    score: float
    verdict: str  # safe | suspicious | phishing
    url_score: float
    reasons: list[str] = field(default_factory=list)
    models: dict = field(default_factory=dict)

    def to_dict(self):
        return {"url": self.url, "score": round(self.score, 3), "verdict": self.verdict,
                "url_score": round(self.url_score, 3), "models": {k: round(v, 3) for k, v in self.models.items()},
                "reasons": self.reasons}


def check(url: str, html: str | None = None, model: Ensemble | None = None) -> Report:
    """Score a URL, and optionally the page behind it. Signals combine with noisy-OR."""
    model = model or default_model()
    probs = model.probs(url)
    p_url = (probs["gbdt"] + probs["ngram"] + probs["lr"]) / 3
    reasons = [f"{n.replace('_', ' ')} (+{c:.1f})" for n, c in model.explain(url)]
    miss = 1 - p_url
    if html is not None:
        for s in analyze(html, url):
            miss *= 1 - s.weight
            reasons.append(f"[{s.id}] {s.note}")
    score = 1 - miss
    t = model.thresholds
    verdict = "phishing" if score >= t["phishing"] else "suspicious" if score >= t["suspicious"] else "safe"
    return Report(url, score, verdict, p_url, reasons, probs)


def check_email(raw, model: Ensemble | None = None) -> EmailReport:
    model = model or default_model()
    return analyze_email(raw, lambda u: check(u, model=model))
