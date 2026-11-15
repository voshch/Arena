"""Recording and drawing robot contacts with held assistive devices (humansim robot_contacts): frames, poses, the start
trigger and the figure."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from arena_runtime.humans.arena_humansim import device_contacts as dc


def _rotate(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> list[float]:
    w, x, y, z = q
    m = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def _xyz(v: tuple[float, float, float]) -> SimpleNamespace:
    return SimpleNamespace(x=v[0], y=v[1], z=v[2])


@pytest.mark.parametrize("d", [(1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (-1.0, -1.0, -0.5), (0.3, -0.2, 1.0)])
def test_quat_toward_turns_x_onto_the_direction_without_roll(d: tuple[float, float, float]) -> None:
    q = dc._quat_toward(*d)
    n = math.sqrt(sum(c * c for c in d))
    assert sum(c * c for c in q) == pytest.approx(1.0)
    assert _rotate(q, (1.0, 0.0, 0.0)) == pytest.approx([c / n for c in d])
    assert _rotate(q, (0.0, 1.0, 0.0))[2] == pytest.approx(0.0)  # the camera's left stays level


@pytest.mark.parametrize(("topic", "frame"), [("/arena/env_0/humansim/robot_contacts", "env_0/map"), ("/arena/env_3/robot_contacts", "env_3/map"), ("/humansim/robot_contacts", "map"), ("/robot_contacts", "map")])
def test_the_env_frame_of_a_topic(topic: str, frame: str) -> None:
    assert dc._env_frame(topic) == frame


def test_a_cane_prop_ends_at_the_tip_moved_into_the_simulator_frame() -> None:
    pytest.importorskip("geometry_msgs")
    d = SimpleNamespace(kind="cane", start=_xyz((1.0, 2.0, 0.9)), end=_xyz((1.6, 2.0, 0.0)), yaw=0.0)
    pose = dc._prop_pose(d, 0.0, 1.2, (5.3, 5.3, 0.0))
    axis = _rotate((pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z), (1.0, 0.0, 0.0))
    top = (pose.position.x, pose.position.y, pose.position.z)
    assert [top[i] + 1.2 * axis[i] for i in range(3)] == pytest.approx([6.9, 7.3, 0.0])
    assert axis == pytest.approx([0.6 / math.hypot(0.6, 0.9), 0.0, -0.9 / math.hypot(0.6, 0.9)])


def test_a_box_prop_sits_at_the_device_origin_its_front_on_the_heading() -> None:
    pytest.importorskip("geometry_msgs")
    d = SimpleNamespace(kind="walking_frame", start=_xyz((3.0, 4.0, 0.05)), end=_xyz((0.0, 0.0, 0.0)), yaw=0.5)
    pose = dc._prop_pose(d, -math.pi / 2, 0.0, (5.3, 5.3, 0.0))
    assert (pose.position.x, pose.position.y, pose.position.z) == pytest.approx((8.3, 9.3, 0.05))
    assert 2 * math.atan2(pose.orientation.z, pose.orientation.w) == pytest.approx(0.5 + math.pi / 2)  # the model faces -y


def test_the_trigger_starts_the_robot_when_a_person_seen_farther_comes_within_reach() -> None:
    trigger, robot = dc._Trigger(2.0), [0.0, 0.0, 0.0]
    assert trigger.update(robot, [{"x": 1.0, "y": 0.0}]) is None  # spawned close: not armed
    assert trigger.update(robot, [{"x": 3.0, "y": 0.0}]) is None
    assert not trigger.fired
    assert trigger.update(robot, [{"x": 9.0, "y": 0.0}, {"x": 1.5, "y": 0.5}]) == pytest.approx(math.hypot(1.5, 0.5))
    assert trigger.fired
    assert trigger.update(robot, [{"x": 1.0, "y": 0.0}]) is None
    assert trigger.fired


def test_without_a_distance_the_trigger_starts_at_once() -> None:
    assert dc._Trigger(0.0).fired
    trigger = dc._Trigger(2.0)
    assert trigger.update(None, [{"x": 9.0, "y": 0.0}]) is None
    assert trigger.update([0.0, 0.0, 0.0], []) is None
    assert not trigger.armed


class _Node:
    def __init__(self) -> None:
        self.logged: list[str] = []

    def create_subscription(self, *args: object) -> None:
        self.callback = args[2]

    def get_logger(self) -> SimpleNamespace:
        return SimpleNamespace(info=self.logged.append, warning=self.logged.append)


def _static(parent: str, child: str, xyz: tuple[float, float, float]) -> SimpleNamespace:
    return SimpleNamespace(transforms=[SimpleNamespace(header=SimpleNamespace(frame_id=parent), child_frame_id=child, transform=SimpleNamespace(translation=_xyz(xyz)))])


def test_the_anchor_waits_for_the_env_tf_then_shifts_by_it() -> None:
    pytest.importorskip("tf2_msgs")
    node = _Node()
    anchor = dc._Anchor(node)
    assert anchor.resolve() is None  # no frame yet
    anchor.frame = "env_0/map"
    assert anchor.resolve() is None  # no TF yet
    node.callback(_static("map", "env_0/map", (5.3, 5.3, 0.0)))
    assert anchor.resolve() == (5.3, 5.3, 0.0)


def test_without_the_anchor_tf_the_env_frame_is_taken_as_the_simulators(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("tf2_msgs")
    import time

    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    node = _Node()
    anchor = dc._Anchor(node)
    anchor.frame = "env_0/map"
    assert anchor.resolve() is None
    now[0] += dc.ANCHOR_WAIT + 1.0
    assert anchor.resolve() == (0.0, 0.0, 0.0)
    assert any("no static TF" in m for m in node.logged)


def test_the_episode_starts_after_the_robot_was_teleported() -> None:
    ticks = [{"t": 0.0, "robot": [-5.3, -5.3, 0.0]}, {"t": 0.1, "robot": [9.0, 11.5, 0.0]}, {"t": 0.2, "robot": [9.04, 11.5, 0.0]}, {"t": 0.3, "robot": None}]
    assert [k["t"] for k in dc._since_reset(ticks)] == [0.1, 0.2, 0.3]
    assert dc._since_reset(ticks[1:]) == ticks[1:]


def _run(tmp_path: Path, part: str) -> Path:
    cane = {"agent_id": 7, "kind": "cane", "start": [6.7, 3.0, 0.9], "end": [6.0, 3.0, 0.0], "centre": [0.0, 0.0, 0.0], "yaw": math.pi, "radius": 0.0125, "size": [0.0, 0.0, 0.0]}
    rows: list[dict[str, Any]] = [{"kind": "meta", "robot_radius": 0.17, "robot_height": 0.35, "drive": 0.4}, {"kind": "walls", "walls": [[0.0, 0.0, 10.0, 0.0], [10.0, 0.0, 10.0, 6.0]]}]
    rows += [{"kind": "tick", "t": 1.0 + 0.1 * i, "robot": [2.0 + 0.04 * i, 3.0, 0.0], "agents": [{"id": 7, "x": 9.0 - 0.07 * i, "y": 3.0, "theta": math.pi, "radius": 0.26}], "devices": [cane]} for i in range(30)]
    edge = {"t": 3.5, "part": part, "robot_id": 1, "robot_name": "jackal", "agent_id": 7, "agent_name": "p", "agent_type": "blind_white_cane_unaware", "point": [6.0, 3.0, 0.1], "depth": 0.01, "robot_pose": [5.8, 3.0, 0.0], "agent_pose": [6.97, 3.0, math.pi]}
    rows.append({"kind": "contact", "event": "start", "duration": 0.0, "device": cane if part == "cane" else None, **edge})
    rows.append({"kind": "contact", "event": "start", "duration": 0.0, "device": cane, **edge, "t": 4.0, "part": "cane"})  # a later contact is not the one drawn
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.mark.parametrize(("part", "title"), [("cane", "Robot contact with a cane (blind_white_cane_unaware)"), ("body", "Robot contact with the person's body (blind_white_cane_unaware)")])
def test_the_figure_draws_the_first_contact_and_the_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], part: str, title: str) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.figure

    figures: list[matplotlib.figure.Figure] = []
    savefig = matplotlib.figure.Figure.savefig
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda self, *a, **k: (figures.append(self), savefig(self, *a, **k))[1])
    out = tmp_path / "run.png"
    assert dc.plot([str(_run(tmp_path, part)), str(out)]) == 0
    assert out.stat().st_size > 0
    assert f"{part} contact at t=2.50 s" in capsys.readouterr().out
    assert title in [t.get_text() for t in figures[0].texts]
    assert "cane" in [t.get_text() for t in figures[0].axes[0].get_legend().get_texts()]  # drawn even when the body is touched first


def test_the_figure_title_fills_in_the_contact(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    assert dc.plot([str(_run(tmp_path, "body")), str(tmp_path / "run.png"), "--title", "Overtaking: {what}, {agent_type}"]) == 0
