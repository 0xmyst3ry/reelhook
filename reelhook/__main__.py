import argparse
import json
import sys

from .data import generate, load_csv
from .detector import check
from .model import LogisticRegression, metrics
from .urlfeatures import FEATURE_NAMES, vector


def cmd_check(a):
    html = open(a.html).read() if a.html else None
    r = check(a.url, html)
    if a.json:
        print(json.dumps(r.to_dict(), indent=2))
    else:
        print(f"{r.verdict.upper()}  score={r.score:.2f}  (url model {r.url_score:.2f})")
        for reason in r.reasons:
            print("  -", reason)
    return 2 if r.verdict == "phishing" else 1 if r.verdict == "suspicious" else 0


def cmd_eval(a):
    train = load_csv(a.train) if a.train else generate(3000, seed=1)
    test = load_csv(a.test) if a.test else generate(1000, seed=99)
    m = LogisticRegression(FEATURE_NAMES).fit([vector(u) for u, _ in train], [l for _, l in train], epochs=60)
    y = [l for _, l in test]
    p = [m.predict_proba(vector(u)) for u, _ in test]
    res = metrics(y, p)
    for k in ("accuracy", "precision", "recall", "f1", "auc"):
        print(f"{k:>9}: {res[k]:.3f}")
    print(f"confusion: tp={res['tp']} fp={res['fp']} fn={res['fn']} tn={res['tn']}")
    print("\nlearned weights (standardised):")
    for n, w in sorted(zip(m.names, m.w), key=lambda t: -abs(t[1]))[:10]:
        print(f"  {n:<18}{w:+.2f}")
    if a.save:
        m.save(a.save)
        print("saved model to", a.save)


def main():
    p = argparse.ArgumentParser(prog="reelhook", description="Phishing URL + page detector")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="score a URL (optionally with its saved HTML)")
    c.add_argument("url")
    c.add_argument("--html", help="path to the page's HTML")
    c.add_argument("--json", action="store_true")
    c.set_defaults(fn=cmd_check)
    e = sub.add_parser("eval", help="train and report held-out metrics")
    e.add_argument("--train", help="CSV url,label")
    e.add_argument("--test", help="CSV url,label")
    e.add_argument("--save", help="write trained model JSON here")
    e.set_defaults(fn=cmd_eval)
    a = p.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
