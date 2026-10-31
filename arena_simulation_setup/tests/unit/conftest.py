from __future__ import annotations

import contextlib
import functools
import http.server
import os
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


@contextlib.contextmanager
def _serving(root: Path) -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def serving() -> Callable[[Path], contextlib._GeneratorContextManager[str]]:
    return _serving


def _open_handles(path: Path) -> int:
    target = os.path.realpath(path)
    count = 0
    for fd in os.listdir("/proc/self/fd"):
        with contextlib.suppress(OSError):
            count += os.readlink(f"/proc/self/fd/{fd}") == target
    return count


@pytest.fixture
def open_handles() -> Callable[[Path], int]:
    return _open_handles
