import argparse
import json
import sys

from .detector import check, check_email


def cmd_check(a):
    html = open(a.html, encoding="utf-8", errors="replace").read() if a.html else None
    r = check(a.url, html)
    if a.json:
        print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"{r.verdict.upper()}  score={r.score:.2f}  (url models {r.url_score:.2f}: "
              + ", ".join(f"{k} {v:.2f}" for k, v in r.models.items()) + ")")
        for reason in r.reasons:
            print("  -", reason)
    return 2 if r.verdict == "phishing" else 1 if r.verdict == "suspicious" else 0


def cmd_email(a):
    raw = open(a.path, "rb").read() if a.path != "-" else sys.stdin.buffer.read()
    r = check_email(raw)
    if a.json:
        print(json.dumps(r.to_dict(), indent=2))
    else:
        print(f"{r.verdict.upper()}  score={r.score:.2f}")
        for s in r.signals:
            print(f"  - [{s.id}] {s.note}")
        for u, ur in r.urls:
            print(f"  link: {ur.verdict:<10} {u[:80]}")
    return 2 if r.verdict == "phishing" else 1 if r.verdict == "suspicious" else 0


def cmd_eval(a):
    from .evaluation import run
    ens, _, _ = run(a.train, a.test)
    if a.save:
        ens.save(a.save)
        print("\nsaved model to", a.save)


def cmd_train(a):
    from .data import load_csv
    from .ensemble import Ensemble
    rows = load_csv(a.train) if a.train else None
    ens = Ensemble.train(rows)
    ens.save()
    print("trained and saved shipped model; thresholds:", {k: round(v, 3) for k, v in ens.thresholds.items()})


def cmd_serve(a):
    from .server import serve
    serve(port=a.port)


def main():
    p = argparse.ArgumentParser(prog="reelhook", description="Phishing detection for URLs, pages and emails")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="score a URL (optionally with its saved HTML)")
    c.add_argument("url")
    c.add_argument("--html", help="path to the page's HTML")
    c.add_argument("--json", action="store_true")
    c.set_defaults(fn=cmd_check)
    m = sub.add_parser("email", help="triage an .eml file ('-' reads stdin)")
    m.add_argument("path")
    m.add_argument("--json", action="store_true")
    m.set_defaults(fn=cmd_email)
    e = sub.add_parser("eval", help="compare models on synthetic and hand-written hold-out data")
    e.add_argument("--train", help="CSV url,label")
    e.add_argument("--test", help="CSV url,label")
    e.add_argument("--save", help="write the trained model JSON here")
    e.set_defaults(fn=cmd_eval)
    t = sub.add_parser("train", help="retrain the shipped model")
    t.add_argument("--train", help="CSV url,label (default: synthetic)")
    t.set_defaults(fn=cmd_train)
    s = sub.add_parser("serve", help="run the HTTP service")
    s.add_argument("--port", type=int, default=8788)
    s.set_defaults(fn=cmd_serve)
    a = p.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
