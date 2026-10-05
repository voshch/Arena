"""Fetch the SELDnet checkpoint and scaler declared in the package's weights.yaml into the data dir."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def data_dir() -> Path:
    return Path(os.environ.get("ARENA_DATA_DIR", "/opt/arena_ws/data")) / "auditory" / "seld"


def manifest() -> list[dict]:
    import yaml
    from ament_index_python.packages import get_package_share_directory

    path = Path(get_package_share_directory("arena_auditory")) / "weights.yaml"
    with open(path) as f:
        return list((yaml.safe_load(f) or {}).get("files", []))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure(root: Path | None = None) -> dict[str, str]:
    """Return ``{role: local path}`` for every manifest entry, downloading what is missing from Hugging Face."""
    root = root or data_dir()
    root.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, str] = {}
    for entry in manifest():
        dest = root / entry["dest"]
        if not dest.is_file():
            from huggingface_hub import hf_hub_download

            cached = Path(hf_hub_download(repo_id=entry["repo"], filename=entry["filename"]))
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_symlink():
                dest.unlink()
            dest.symlink_to(cached)
        actual = _sha256(dest)
        if actual != entry["sha256"]:
            raise RuntimeError(f"{dest}: sha256 {actual} does not match weights.yaml ({entry['sha256']})")
        resolved[entry["role"]] = str(dest)
    return resolved


def main() -> None:
    for role, path in ensure().items():
        print(f"{role}: {path}")
