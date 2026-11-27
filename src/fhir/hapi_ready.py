from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from typing import Callable


def metadata_ok(url: str, timeout: float = 2) -> bool:
    """Return whether GET url responds with a 2xx status."""

    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= int(response.status) < 300
    except Exception:
        return False


def wait_until_ready(url: str, timeout_s: float, poll_s: float = 1) -> bool:
    """Poll metadata until it answers or `timeout_s` elapses."""

    deadline = time.perf_counter() + timeout_s
    while True:
        if metadata_ok(url):
            return True
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return False
        time.sleep(min(poll_s, remaining))


def recover_ready(
    wait_ready: Callable[[float], bool],
    restart: Callable[[], None],
    timeout_s: float,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Wait for HAPI. On timeout, restart once and wait again."""

    started = clock()
    if wait_ready(timeout_s):
        return _report(True, False, started, 0, 1, clock)
    restart_started = clock()
    restart()
    restart_ms = int((clock() - restart_started) * 1000)
    ready = wait_ready(timeout_s)
    return _report(ready, True, started, restart_ms, 2, clock)


def _report(ready: bool, restarted: bool, started: float, restart_ms: int, attempts: int, clock) -> dict:
    return {
        "ready": ready,
        "restarted": restarted,
        "ready_ms": int((clock() - started) * 1000),
        "restart_ms": restart_ms,
        "attempts": attempts,
    }


def _restart_container(docker: str, container: str) -> None:
    """Restart one container. `docker` is the executable, possibly with a sudo prefix."""

    command = [*docker.split(), "restart", container]
    subprocess.run(command, check=False)


def main(argv: list[str] | None = None) -> int:
    """Wait for metadata, restart the HAPI container once if that times out."""

    parser = argparse.ArgumentParser(description="Wait until HAPI answers metadata.")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--metadata-url", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--poll", type=float, default=1.0)
    args = parser.parse_args(argv)

    def wait(timeout_s: float) -> bool:
        return wait_until_ready(args.metadata_url, timeout_s, poll_s=args.poll)

    def restart() -> None:
        print(
            f"HAPI not ready after {args.timeout:g}s; restarting container {args.container}",
            file=sys.stderr,
        )
        _restart_container(args.docker, args.container)

    report = recover_ready(wait, restart, args.timeout)
    print("READY_JSON " + json.dumps(report))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
