import json
import socket
import threading
import time
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
        server.server_close()


class _SlowBodyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "32")
        self.end_headers()
        self.wfile.flush()
        time.sleep(1.5)
        try:
            self.wfile.write(b'{"resourceType":"Observation"}')
        except OSError:
            pass

    def log_message(self, fmt, *args):
        return


def _serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_read_timeout_returns_status_zero():
    server = _serve(_SlowBodyHandler)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/Observation/1"
        _, call = ExecHarness("http://127.0.0.1/", http=UrllibHttpClient(timeout=0.3)).step(
            AgentAction(kind="GET", url=url)
        )
    finally:
        server.shutdown()
        server.server_close()
    assert call.status == 0
    assert call.response_body["error"]


def test_dropped_connection_returns_status_zero():
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def drop():
        conn, _ = listener.accept()
        try:
            conn.recv(65536)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\n")
        finally:
            conn.close()
            listener.close()

    threading.Thread(target=drop, daemon=True).start()
    url = f"http://127.0.0.1:{port}/Observation/1"
    _, call = ExecHarness("http://127.0.0.1/", http=UrllibHttpClient(timeout=2)).step(
        AgentAction(kind="GET", url=url)
    )
    assert call.status == 0
    assert call.response_body["error"]
