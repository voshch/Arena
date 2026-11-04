#!/bin/bash
set -eo pipefail
uv lock --project .uv-workspace
torch=$(python3 -c 'import tomllib; print(*(p["version"] for p in tomllib.load(open(".uv-workspace/uv.lock", "rb"))["package"] if p["name"] == "torch"))')
mapfile -t skipped < <(python3 -c 'import re, tomllib; [print(p["name"]) for p in tomllib.load(open(".uv-workspace/uv.lock", "rb"))["package"] if re.fullmatch("torch|triton|cuda-.*|nvidia-.*", p["name"])]')
uv sync --inexact --active --project .uv-workspace $(printf -- '--no-install-package %s ' "${skipped[@]}") "$@"
uv pip install --no-deps --index-url https://download.pytorch.org/whl/cpu "torch==$torch"
