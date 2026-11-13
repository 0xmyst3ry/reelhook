import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .detector import check, check_email

MAX_BODY = 5_000_000


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, {"status": "ok"}) if self.path == "/health" else self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY:
                return self._send(413, {"error": "payload too large"})
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/check":
                url, html = body["url"], body.get("html")
                if not isinstance(url, str) or (html is not None and not isinstance(html, str)):
                    raise ValueError
                return self._send(200, check(url, html).to_dict())
            if self.path == "/email":
                raw = body["raw"]
                if not isinstance(raw, str):
                    raise ValueError
                return self._send(200, check_email(raw).to_dict())
            self._send(404, {"error": "not found"})
        except (ValueError, KeyError, json.JSONDecodeError):
            self._send(400, {"error": 'expected JSON {"url": "...", "html": "..."?} or {"raw": "<rfc822 email>"}'})

    def log_message(self, *a):
        pass


def serve(host="127.0.0.1", port=8788):
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"reelhook listening on http://{host}:{port}  (POST /check, /email)")
    srv.serve_forever()
