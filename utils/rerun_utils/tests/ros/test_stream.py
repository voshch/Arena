"""What the bridge logs and when: streams on sim_time, one-time data static, rate-capped per key, payloads shrunk."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import rerun as rr
from arena_viz import DisplayKind
from builtin_interfaces.msg import Time
from PIL import Image as PILImage
from rerun.chunk import RrdReader
from sensor_msgs.msg import Image

from rerun_utils.blueprint import blueprint
from rerun_utils.renderers.image import image_archetype
from rerun_utils.renderers.points3d import thinned
from rerun_utils.stream import TIMELINE, Stream
from rerun_utils.tf_mirror import TransformMemo


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _recording(path: Path) -> rr.RecordingStream:
    rec = rr.RecordingStream("test_stream", recording_id=path.stem)
    rec.save(str(path))
    return rec


def _chunks(rec: rr.RecordingStream, path: Path) -> dict[str, list[tuple[bool, list[str]]]]:
    rec.flush()
    rec.disconnect()
    out: dict[str, list[tuple[bool, list[str]]]] = {}
    for chunk in RrdReader(path).stream():
        out.setdefault(chunk.entity_path, []).append((chunk.is_static, chunk.timeline_names))
    return out


def _blob(archetype: rr.EncodedImage | rr.EncodedDepthImage) -> bytes:
    return archetype.blob.as_arrow_array()[0].values.to_numpy().tobytes()


def test_streams_ride_sim_time_and_one_time_data_is_static(tmp_path: Path) -> None:
    path = tmp_path / "stream.rrd"
    rec = _recording(path)
    stream = Stream(rec, max_rate_hz=0.0, sim_now=lambda: 4.0)
    stream.log("/scan", rr.Points3D([[1.0, 0.0, 0.0]]), stamp=Time(sec=2))
    stream.log("/scan", rr.Points3D([[2.0, 0.0, 0.0]]))
    stream.fixed("/map", rr.Points3D([[0.0, 0.0, 0.0]]))
    chunks = _chunks(rec, path)
    assert chunks["/scan"] and all(not static and TIMELINE in timelines for static, timelines in chunks["/scan"])
    assert all(static for static, _ in chunks["/map"])


def test_an_early_offer_waits_and_only_the_newest_runs_once_its_slot_comes(tmp_path: Path) -> None:
    clock = Clock()
    stream = Stream(_recording(tmp_path / "rate.rrd"), max_rate_hz=10.0, sim_now=lambda: 0.0, clock=clock)
    ran: list[str] = []
    stream.offer("scan", lambda: ran.append("first"))
    clock.now = 0.02
    stream.offer("scan", lambda: ran.append("second"))
    stream.offer("scan", lambda: ran.append("third"))
    stream.offer("odom", lambda: ran.append("odom"))
    stream.flush()
    assert ran == ["first", "odom"]
    clock.now = 0.1
    stream.flush()
    stream.flush()
    assert ran == ["first", "odom", "third"]


def test_an_uncapped_stream_runs_every_offer(tmp_path: Path) -> None:
    stream = Stream(_recording(tmp_path / "uncapped.rrd"), max_rate_hz=0.0, sim_now=lambda: 0.0, clock=Clock())
    ran: list[int] = []
    for i in range(3):
        stream.offer("scan", lambda i=i: ran.append(i))
    assert ran == [0, 1, 2]


def test_a_repeated_transform_is_unchanged_until_it_moves_or_changes_parent() -> None:
    memo = TransformMemo()
    pose = (1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    assert memo.changed("base_link", "odom", pose)
    assert not memo.changed("base_link", "odom", (1.0 + 1e-7, *pose[1:]))
    assert memo.changed("base_link", "odom", (1.01, *pose[1:]))
    assert memo.changed("base_link", "map", (1.01, *pose[1:]))


def _frame(encoding: str, pixel: tuple[int, ...]) -> Image:
    return Image(height=2, width=2, encoding=encoding, data=bytes(pixel * 4))


def test_color_frames_become_jpeg_in_rgb_order() -> None:
    for encoding, pixel in (("rgb8", (255, 0, 0)), ("bgr8", (0, 0, 255)), ("bgra8", (0, 0, 255, 255)), ("rgba8", (255, 0, 0, 255))):
        archetype = image_archetype(_frame(encoding, pixel))
        assert isinstance(archetype, rr.EncodedImage), encoding
        red, green, blue = np.asarray(PILImage.open(io.BytesIO(_blob(archetype))).convert("RGB"))[0, 0]
        assert red > 200 and green < 60 and blue < 60, encoding


def test_depth_frames_become_millimeter_pngs_and_unknown_encodings_are_skipped() -> None:
    meters = np.array([1.5, np.nan, 0.25, 100.0], np.float32)
    archetype = image_archetype(Image(height=2, width=2, encoding="32FC1", data=meters.tobytes()))
    assert isinstance(archetype, rr.EncodedDepthImage)
    assert np.asarray(PILImage.open(io.BytesIO(_blob(archetype)))).ravel().tolist() == [1500, 0, 250, 65535]
    assert image_archetype(Image(height=1, width=1, encoding="yuv422", data=b"\0\0")) is None


def test_clouds_keep_their_finite_points_strided_under_the_cap() -> None:
    pts = np.arange(30, dtype=np.float32).reshape(10, 3)
    pts[3, 1] = np.nan
    assert len(thinned(pts, cap=100)) == 9
    assert len(thinned(np.zeros((1001, 3), np.float32), cap=100)) <= 100


def test_the_layout_adds_side_views_only_for_cameras_and_series() -> None:
    displays = [
        (DisplayKind.LASER_SCAN, "env_0/robots/r/scan", "r/scan"),
        (DisplayKind.IMAGE, "env_0/robots/r/cam", "r/cam"),
        (DisplayKind.IMU, "env_0/robots/r/imu", "r/imu"),
    ]
    root = blueprint(displays).root_container
    assert type(root).__name__ == "Horizontal"
    side = root.contents[1]
    assert [type(view).__name__ for view in side.contents] == ["Spatial2DView", "TimeSeriesView"]
    assert [type(view).__name__ for view in blueprint(displays[:1]).root_container.contents] == ["Spatial3DView"]
