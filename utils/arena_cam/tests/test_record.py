from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from arena_cam.record import Recorder, claim, default_name, env_take_path, final_path, promote, record_path, record_root, tagged, take_paths

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


def test_record_path_defaults_and_tags_per_camera(tmp_path: Path) -> None:
    path = record_path(str(tmp_path / "takes" / "tour"))
    assert path == tmp_path / "takes" / "tour.mp4"
    assert path.parent.is_dir()
    assert record_path(str(tmp_path / "tour.mkv")).suffix == ".mkv"
    assert tagged(path, "sim") == tmp_path / "takes" / "tour-sim.mp4"
    assert tagged(path, "viz0") == tmp_path / "takes" / "tour-viz0.mp4"
    assert take_paths(path, ["viz0"]) == [path]
    assert take_paths(path, ["sim", "viz0"]) == [tagged(path, "sim"), tagged(path, "viz0")]


def test_claim_refuses_clobber(tmp_path: Path) -> None:
    taken, free = tmp_path / "tour-sim.mp4", tmp_path / "tour-viz0.mp4"
    taken.write_bytes(b"")
    claim([free])
    with pytest.raises(FileExistsError, match="tour-sim.mp4"):
        claim([taken, free])
    claim([taken, free], force=True)


def test_default_name_is_stem_and_filesystem_safe_timestamp() -> None:
    assert re.fullmatch(r"orbit_\d{8}-\d{6}", default_name("orbit"))


def test_a_relative_record_dir_lands_under_recordings(tmp_path: Path) -> None:
    assert record_root("mvp/run1", tmp_path) == tmp_path / "recordings" / "mvp" / "run1"


def test_an_absolute_record_dir_is_kept(tmp_path: Path) -> None:
    assert record_root(str(tmp_path / "x"), Path("/elsewhere")) == tmp_path / "x"


def test_a_take_lands_in_its_env_directory(tmp_path: Path) -> None:
    assert env_take_path(tmp_path, 3, "w/s/ep001") == tmp_path / "env_3" / "w" / "s" / "ep001.mp4"


def test_two_envs_recording_the_same_label_keep_their_own_files(tmp_path: Path) -> None:
    assert env_take_path(tmp_path, 0, "w/ep001") != env_take_path(tmp_path, 1, "w/ep001")


def test_a_partial_view_file_maps_to_its_final_name() -> None:
    assert final_path(Path("w/s/ep001.partial-pedestrian.mp4")) == Path("w/s/ep001-pedestrian.mp4")


def test_a_lone_camera_partial_maps_to_the_bare_take() -> None:
    assert final_path(Path("w/ep001.partial.mp4")) == Path("w/ep001.mp4")


def test_promote_renames_every_view(tmp_path: Path) -> None:
    files = [tmp_path / f"ep001.partial-{view}.mp4" for view in ("a", "b")]
    for file in files:
        file.write_bytes(b"x")
    assert promote(files) == [tmp_path / "ep001-a.mp4", tmp_path / "ep001-b.mp4"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["ep001-a.mp4", "ep001-b.mp4"]


def test_promote_skips_a_view_that_never_got_written(tmp_path: Path) -> None:
    (tmp_path / "ep001.partial-a.mp4").write_bytes(b"x")
    assert promote([tmp_path / "ep001.partial-a.mp4", tmp_path / "ep001.partial-b.mp4"]) == [tmp_path / "ep001-a.mp4"]


def test_promote_never_overwrites_a_finished_take(tmp_path: Path) -> None:
    (tmp_path / "ep001-a.mp4").write_bytes(b"old")
    (tmp_path / "ep001.partial-a.mp4").write_bytes(b"new")
    assert promote([tmp_path / "ep001.partial-a.mp4"]) == []
    assert (tmp_path / "ep001-a.mp4").read_bytes() == b"old"
