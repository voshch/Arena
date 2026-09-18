from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from arena_cam.record import Recorder, default_name, record_path

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not on PATH")


def _frame(width: int, height: int, n: int) -> bytes:
    return bytes((x + n * 8) % 256 for _ in range(height) for x in range(width) for _ in range(3))


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate,pix_fmt", "-of", "json", str(path)],
        check=True,
        capture_output=True,
    )
    return json.loads(out.stdout)["streams"][0]


def test_encodes_every_frame(tmp_path: Path) -> None:
    recorder = Recorder(str(tmp_path / "take.mp4"), 24)
    for n in range(12):
        recorder.write_rgb(64, 48, _frame(64, 48, n))
    assert recorder.close()
    stream = _probe(recorder.path)
    assert (stream["width"], stream["height"]) == (64, 48)
    assert int(stream["nb_read_frames"]) == 12
    assert stream["r_frame_rate"] == "24/1"
    assert stream["pix_fmt"] == "yuv420p"


def test_odd_viewport_is_padded_even(tmp_path: Path) -> None:
    recorder = Recorder(str(tmp_path / "odd.mp4"), 30)
    for n in range(3):
        recorder.write_rgb(63, 47, _frame(63, 47, n))
    assert recorder.close()
    stream = _probe(recorder.path)
    assert (stream["width"], stream["height"]) == (64, 48)


def test_size_change_mid_take_is_refused(tmp_path: Path) -> None:
    recorder = Recorder(str(tmp_path / "resize.mp4"), 30)
    recorder.write_rgb(64, 48, _frame(64, 48, 0))
    with pytest.raises(ValueError, match="frame size changed"):
        recorder.write_rgb(32, 24, _frame(32, 24, 1))
    assert recorder.close()


def test_no_frames_is_a_failed_take(tmp_path: Path) -> None:
    assert not Recorder(str(tmp_path / "empty.mp4"), 30).close()


def test_record_path_defaults_and_refuses_clobber(tmp_path: Path) -> None:
    path = record_path(str(tmp_path / "takes" / "tour"))
    assert path == tmp_path / "takes" / "tour.mp4"
    path.write_bytes(b"")
    with pytest.raises(FileExistsError):
        record_path(str(path))
    assert record_path(str(path), force=True) == path
    assert record_path(str(tmp_path / "tour.mkv")).suffix == ".mkv"


def test_default_name_is_stem_and_filesystem_safe_timestamp() -> None:
    assert re.fullmatch(r"orbit_\d{8}-\d{6}", default_name("orbit"))
