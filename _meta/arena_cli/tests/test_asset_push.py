"""arena asset push up to the upload: file selection against real clips in a temporary local root, --dry-run stops before the bucket."""

import os
import subprocess
import sys
from pathlib import Path


def _clip(root: Path, name: str) -> Path:
    clip = root / "Common" / "Animation" / name
    clip.mkdir(parents=True)
    for file in (f"{name}.npz", f"{name}.anim", "annotation.yaml", ".ttl"):
        (clip / file).write_text("x")
    return clip


def _push(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "ARENA_ASSETS_DIR": str(root / "cache"),
        "ARENA_ASSETS_DIR_LOCAL": str(root),
        "PYTHONPATH": os.pathsep.join([str(Path(__file__).parents[2]), os.environ.get("PYTHONPATH", "")]),
    }
    return subprocess.run([sys.executable, "-m", "arena_cli", "asset", "push", *args], env=env, capture_output=True, text=True, check=False)


def test_only_keeps_the_matching_files_of_one_asset(tmp_path):
    _clip(tmp_path, "wave")
    done = _push(tmp_path, "animation", "wave", "--only", "*.anim", "--dry-run")
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[0] == "Common/Animation/wave/wave.anim"
    assert "would publish 1 file(s)" in done.stdout


def test_all_stages_every_local_asset_for_one_upload(tmp_path):
    _clip(tmp_path, "wave")
    _clip(tmp_path, "hug")
    done = _push(tmp_path, "animation", "--all", "--only", "*.anim,*.yaml", "--dry-run")
    assert done.returncode == 0, done.stderr
    listed = done.stdout.splitlines()[:-1]
    assert listed == ["Common/Animation/hug/annotation.yaml", "Common/Animation/hug/hug.anim", "Common/Animation/wave/annotation.yaml", "Common/Animation/wave/wave.anim"]


def test_cache_stamp_never_ships(tmp_path):
    _clip(tmp_path, "wave")
    done = _push(tmp_path, "animation", "wave", "--dry-run")
    assert done.returncode == 0, done.stderr
    assert ".ttl" not in done.stdout and "wave.npz" in done.stdout


def test_a_glob_matching_nothing_publishes_nothing(tmp_path):
    _clip(tmp_path, "wave")
    done = _push(tmp_path, "animation", "wave", "--only", "*.mp4", "--dry-run")
    assert done.returncode == 1 and "nothing to publish" in done.stderr


def test_name_and_all_exclude_each_other(tmp_path):
    _clip(tmp_path, "wave")
    for args in (("animation", "wave", "--all"), ("animation",)):
        done = _push(tmp_path, *args)
        assert done.returncode != 0 and "one animation name or --all" in done.stderr + done.stdout


def test_only_refuses_a_kind_that_publishes_one_file(tmp_path):
    done = _push(tmp_path, "suite", "any", "--only", "*.yaml", "--dry-run")
    assert done.returncode != 0 and "--only filters directory assets" in done.stderr + done.stdout
