#!/usr/bin/env python3
"""Thin model-manager daemon for vla_server.

Holds no model itself. On `POST /ensure {model}` it spawns that model's runner (the server script and
weights from models.yaml) on a free port if one is not already running, and returns the port. Runners
share the daemon's host network, so the caller connects to `127.0.0.1:<port>`.
"""

import os
import socket
import subprocess
import threading
import time
import urllib.request

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = "/opt/arena_ws/data/vla"

with open(os.path.join(_DIR, "models.yaml")) as f:
    _MODELS = yaml.safe_load(f)

app = FastAPI()
_lock = threading.Lock()
_runners: dict[str, tuple[subprocess.Popen, int]] = {}


class EnsureRequest(BaseModel):
    model: str


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
            return r.status == 200
    except OSError:
        return False


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ensure")
def ensure(req: EnsureRequest) -> dict:
    if req.model not in _MODELS:
        raise HTTPException(status_code=404, detail=f"unknown model {req.model!r}")
    with _lock:
        existing = _runners.get(req.model)
        if existing is not None and existing[0].poll() is None:
            return {"port": existing[1]}
        row = _MODELS[req.model]
        port = _free_port()
        weights = os.path.join(_DATA_DIR, req.model, row["weights"])
        proc = subprocess.Popen(["python3", os.path.join(_DIR, row["server"]), "--port", str(port), "--weights", weights])
        for _ in range(180):
            if _healthy(port):
                _runners[req.model] = (proc, port)
                return {"port": port}
            if proc.poll() is not None:
                raise HTTPException(status_code=500, detail=f"runner for {req.model!r} exited with {proc.returncode}")
            time.sleep(1)
        proc.terminate()
        raise HTTPException(status_code=504, detail=f"runner for {req.model!r} did not become healthy")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
