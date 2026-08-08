"""Tiny stdlib HTTP app used by P-3 live-probe tests.

Mode is chosen by ``FIXTURE_MODE`` so one script covers every scenario the
probe test suite needs (satisfying/violating fixtures per probe), matching
how the P-2 static-probe tests use one fixture-per-outcome without needing
a separate real framework install per case.

Modes:
  graceful    - normal app: /health always 200, /ready reflects
                FIXTURE_READY, exits cleanly on SIGTERM.
  lying_ready - /ready aliases /health byte-for-byte (ignores dependency
                state) — the readiness-distinctness violation.
  no_ready    - no /ready route at all (404) — the other readiness
                violation shape.
  stubborn    - ignores SIGTERM entirely — the graceful-shutdown violation.
"""

from __future__ import annotations

import os
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_MODE = os.environ.get("FIXTURE_MODE", "graceful")
_shutdown = threading.Event()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence stderr spam
        pass

    def do_GET(self):
        if self.path == "/health":
            self._reply(200, b"ok")
        elif self.path == "/ready":
            if _MODE == "no_ready":
                self._reply(404, b"not found")
            elif _MODE == "lying_ready":
                self._reply(200, b"ok")  # byte-identical to /health: aliasing
            else:
                ready = os.environ.get("FIXTURE_READY", "1") == "1"
                self._reply(200 if ready else 503, b"ready" if ready else b"not ready")
        else:
            self._reply(200, b"hello")

    def _reply(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _on_term(signum, frame):
    _shutdown.set()


def main() -> None:
    port = int(os.environ["PORT"])
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = False

    if _MODE == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGTERM, _on_term)

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    if _MODE == "stubborn":
        t.join()  # never stops on SIGTERM; only a SIGKILL ends this process
    else:
        _shutdown.wait()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
