import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from harness.exec_harness import ExecHarness
from harness.http_client import UrllibHttpClient
from harness.types import AgentAction


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        body = json.loads(raw)
        body["id"] = "local-1"
        encoded = json.dumps(body).encode()
        self.send_response(201)
        self.send_header("Content-Type", "application/fhir+json")
        self.send_header("Location", f"http://127.0.0.1:{self.server.server_address[1]}/Observation/local-1/_history/1")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_PUT(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/fhir+json")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt, *args):
        return


def test_urllib_client_posts_and_puts_against_a_local_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}/"
        harness = ExecHarness(base, http=UrllibHttpClient())
        created, call = harness.step(
            AgentAction(kind="POST", url=base + "Observation", body={"resourceType": "Observation", "status": "final"})
        )
        assert call.status == 201
        assert created.location.endswith("/Observation/local-1/_history/1")
        updated, put_call = harness.step(
            AgentAction(
                kind="PUT",
                url=base + "Observation/local-1",
                body={"resourceType": "Observation", "id": "local-1", "status": "final"},
            )
        )
        assert put_call.status == 200
        assert updated.body["id"] == "local-1"
    finally:
        server.shutdown()
