#!/usr/bin/env python3
"""
Finalize one Arena acoustic scenario, upload it to Hugging Face, verify it,
and optionally delete the local scenario directory.

Required local products after arena_simulation_setup.acoustics.export_recording:
  - one MCAP
  - one 4-channel WAV/FLAC
  - one 2-channel WAV/FLAC
  - pedestrian trajectory parquet
  - robot trajectory parquet
  - labels/annotations (json/jsonl/parquet)
  - validation json

The original MCAP is never transcoded or rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError


def die(msg: str) -> None:
    raise SystemExit(f"finalize_acoustic_scenario: ERROR: {msg}")


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def audio_channels(path: Path) -> int | None:
    """Read channel count without requiring external multimedia binaries."""
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wav:
                return int(wav.getnchannels())
        except (wave.Error, EOFError):
            # IEEE-float WAV is valid but unsupported by some Python `wave`
            # versions. SciPy is already part of the Arena runtime.
            try:
                from scipy.io import wavfile

                _, samples = wavfile.read(path, mmap=True)
                return 1 if samples.ndim == 1 else int(samples.shape[1])
            except Exception:
                return None
    # Canonical files created by older successful finalizations are named by
    # stream role, so they remain discoverable without ffprobe.
    if path.name == "raw.flac":
        return 4
    if path.name == "rendered.flac":
        return 2
    return None


def choose_audio(run_dir: Path, channels: int) -> Path:
    candidates: list[Path] = []
    for p in run_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".wav", ".flac"}:
            continue
        if p.name in {"raw.flac", "rendered.flac"}:
            continue
        if audio_channels(p) == channels:
            candidates.append(p)
    if not candidates:
        die(f"no {channels}-channel WAV/FLAC found under {run_dir}")
    # Exporters sometimes leave short diagnostics clips. Prefer the real recording.
    return max(candidates, key=lambda p: p.stat().st_size)


def choose_unique(run_dir: Path, patterns: Iterable[str], description: str) -> Path:
    matches: list[Path] = []
    for pat in patterns:
        matches.extend(p for p in run_dir.rglob(pat) if p.is_file())
    # de-duplicate while preserving paths
    matches = list(dict.fromkeys(matches))
    if not matches:
        die(f"missing {description} under {run_dir}")
    if len(matches) == 1:
        return matches[0]
    # Prefer explicit/canonical names, otherwise the largest artifact.
    return max(matches, key=lambda p: p.stat().st_size)


def collect_labels(run_dir: Path) -> list[Path]:
    keywords = ("label", "annotation", "activity", "sound_event", "overlap", "source")
    found = []
    for p in run_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".json", ".jsonl", ".parquet", ".csv"}:
            continue
        low = p.name.lower()
        if any(k in low for k in keywords) and "validation" not in low and p.name != "manifest.json":
            found.append(p)
    if not found:
        die("no label/annotation artifact found")
    return sorted(set(found))


def write_checksums(run_dir: Path) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    excluded = {"checksums.sha256", "upload_receipt.json", "hf_upload.log", "launch.log"}
    for p in sorted(x for x in run_dir.rglob("*") if x.is_file()):
        rel = p.relative_to(run_dir).as_posix()
        if rel in excluded:
            continue
        inventory[rel] = {
            "sha256": sha256(p),
            "size": p.stat().st_size,
        }
    checksum_path = run_dir / "checksums.sha256"
    checksum_path.write_text(
        "".join(f"{v['sha256']}  {name}\n" for name, v in inventory.items()),
        encoding="utf-8",
    )
    inventory["checksums.sha256"] = {
        "sha256": sha256(checksum_path),
        "size": checksum_path.stat().st_size,
    }
    return inventory


def verify_remote(
    api: HfApi,
    repo_id: str,
    remote_prefix: str,
    inventory: dict[str, dict[str, object]],
) -> None:
    paths = [f"{remote_prefix}/{rel}" for rel in inventory]
    infos = api.get_paths_info(repo_id, paths=paths, repo_type="dataset")
    by_path = {x.path: x for x in infos}

    missing = [p for p in paths if p not in by_path]
    if missing:
        die("remote verification failed; missing: " + ", ".join(missing))

    for rel, local in inventory.items():
        remote_path = f"{remote_prefix}/{rel}"
        info = by_path[remote_path]
        if int(info.size) != int(local["size"]):
            die(f"remote size mismatch for {remote_path}: local={local['size']} remote={info.size}")

        # Large Hub objects normally expose LFS SHA-256. When it is present,
        # require an exact content hash match. For small regular Git blobs,
        # size + successful committed-path lookup is the available metadata
        # check without downloading the file back.
        if info.lfs is not None:
            remote_sha = info.lfs.sha256
            if remote_sha and remote_sha != local["sha256"]:
                die(f"remote SHA-256 mismatch for {remote_path}: local={local['sha256']} remote={remote_sha}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, nargs="?")
    ap.add_argument("--repo-id", default="kameow/arena_audio")
    ap.add_argument("--remote-prefix", required=True)
    ap.add_argument("--token-env", default="HF_TOKEN")
    visibility = ap.add_mutually_exclusive_group()
    visibility.add_argument("--private", dest="private", action="store_true", default=True)
    visibility.add_argument("--public", dest="private", action="store_false")
    ap.add_argument("--check-remote", action="store_true")
    ap.add_argument("--delete-after-verify", action="store_true")
    ap.add_argument("--world-name", required=True)
    ap.add_argument("--scenario-name", required=True)
    ap.add_argument("--execution-index", type=int, required=True)
    args = ap.parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        die(f"{args.token_env} is not set. Use a Hugging Face write token, e.g. `export {args.token_env}=hf_...` before recording.")

    api = HfApi(token=token)
    if args.check_remote:
        required = [
            f"{args.remote_prefix}/manifest.json",
            f"{args.remote_prefix}/checksums.sha256",
        ]
        try:
            infos = api.get_paths_info(args.repo_id, paths=required, repo_type="dataset")
        except RepositoryNotFoundError:
            print("MISSING")
            return 0
        found = {item.path for item in infos}
        print("EXISTS" if all(path in found for path in required) else "MISSING")
        return 0

    if args.run_dir is None:
        die("run_dir is required unless --check-remote is used")
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        die(f"run directory not found: {run_dir}")
    validations = sorted(run_dir.glob("*_validation.json"))
    if not validations:
        die("validation JSON is missing")
    validation = json.loads(validations[0].read_text())
    if validation.get("valid") is not True:
        die(f"scenario validation is not valid: {validations[0]}")

    # Canonical storage contract.
    mcap = choose_unique(run_dir, ["*.mcap", "**/*.mcap"], "MCAP")

    raw_src = choose_audio(run_dir, 4)
    rendered_src = choose_audio(run_dir, 2)

    ped_src = choose_unique(
        run_dir,
        ["*pedestrian*.parquet", "*human*.parquet", "**/*pedestrian*.parquet", "**/*human*.parquet"],
        "pedestrian trajectory Parquet",
    )
    robot_src = choose_unique(
        run_dir,
        ["*robot*.parquet", "*jackal*.parquet", "**/*robot*.parquet", "**/*jackal*.parquet"],
        "robot trajectory Parquet",
    )
    ped = ped_src
    robot = robot_src

    labels = collect_labels(run_dir)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "world": args.world_name,
        "scenario": args.scenario_name,
        "execution_index": args.execution_index,
        "authoritative_source": mcap.relative_to(run_dir).as_posix(),
        "model_input_audio": {
            "path": raw_src.relative_to(run_dir).as_posix(),
            "channels": 4,
            "encoding": raw_src.suffix.lstrip(".").upper(),
            "lossless": True,
        },
        "monitor_audio": {
            "path": rendered_src.relative_to(run_dir).as_posix(),
            "channels": 2,
            "encoding": rendered_src.suffix.lstrip(".").upper(),
            "lossless": True,
            "optional_for_training": True,
        },
        "trajectories": {
            "pedestrians": ped.relative_to(run_dir).as_posix(),
            "robot": robot.relative_to(run_dir).as_posix(),
        },
        "labels": [p.relative_to(run_dir).as_posix() for p in labels],
        "validation": validations[0].relative_to(run_dir).as_posix(),
        "preserved_from_mcap": [
            "/clock and audio timing",
            "microphone geometry and channel mapping",
            "sound activity and source identity",
            "motor/self-noise and overlap labels",
            "TF",
            "map geometry",
            "episode metadata",
        ],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    inventory = write_checksums(run_dir)

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(run_dir),
        path_in_repo=args.remote_prefix,
        allow_patterns=list(inventory),
        commit_message=(f"Add acoustic scenario {args.world_name}/{args.scenario_name} #{args.execution_index:04d}"),
    )

    verify_remote(api, args.repo_id, args.remote_prefix, inventory)

    receipt = {
        "repo_id": args.repo_id,
        "remote_prefix": args.remote_prefix,
        "commit": commit.oid,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "files_verified": len(inventory),
    }
    (run_dir / "upload_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(receipt, sort_keys=True))

    if args.delete_after_verify:
        # Keep nothing local once all required remote objects have been verified.
        shutil.rmtree(run_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
