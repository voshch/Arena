"""Humansim world objects: the pose the engine queues people from, built from the model's annotation and the scenario's `humansim:` block."""

from __future__ import annotations

import asyncio
import math
import types
import typing
from pathlib import Path

import pytest

pytest.importorskip("rclpy")

from arena_runtime.env.realizer import Realizer
from arena_runtime.humans.arena_humansim.arena_humansim import ArenaHumanSimulator
from arena_runtime.shared import Obstacle, Orientation, Pose, Position

YAW = math.pi / 2
BOX = "bounding_box: [[-0.5, 0.5], [-0.5, 0.5], [0.0, 1.0]]\n"


class _Client:
    def __init__(self) -> None:
        self.requests: list[typing.Any] = []

    async def call_timeout(self, request: typing.Any) -> typing.Any:
        self.requests.append(request)
        return types.SimpleNamespace(success=True, message="")


class _Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def get_child(self, _name: str) -> _Logger:
        return self

    def __getattr__(self, level: str) -> typing.Callable[..., None]:
        return lambda msg, *a, **k: self.lines.append(f"{level}: {msg}")


class _Node:
    def __init__(self) -> None:
        self.logger = _Logger()

    def get_logger(self) -> _Logger:
        return self.logger


def _adapter() -> ArenaHumanSimulator:
    sim = ArenaHumanSimulator.__new__(ArenaHumanSimulator)
    sim._NodeInterface__node = _Node()
    sim._realizer = Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix="env_0"), {})
    sim._add_obstacles_client = _Client()
    sim._add_world_objects_client = _Client()
    sim._seats = {}
    sim._seat_slots = {}
    sim._warned_faces = set()
    return sim


def _spawn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, files: dict[str, str | None], humansim: dict | None = None, names: tuple[str, ...] = ("desk",)) -> tuple[ArenaHumanSimulator, list[typing.Any]]:
    sim = _adapter()
    obstacles = []
    for name in names:
        obstacle = Obstacle(name=name, model="desk", pose=Pose(Position(1.0, 2.0, 0.0), Orientation.from_yaw(YAW)), humansim=humansim or {})
        obstacle.sim_path = f"env_0/{name}"
        obstacles.append(obstacle)
    bundle = tmp_path / "desk"
    bundle.mkdir()
    for file, text in files.items():
        if text is not None:
            (bundle / file).write_text(text)

    async def resolve_path(_self: object) -> Path:
        return bundle

    monkeypatch.setattr(type(obstacles[0].model), "resolve_path", resolve_path)
    asyncio.run(sim._spawn_obstacles_impl(obstacles))
    (request,) = sim._add_world_objects_client.requests
    return sim, list(request.objects)


def _queue_side(pose: typing.Any) -> tuple[float, float]:
    """Where humansim puts the first person in line, one metre out: behind theta, as its line formation does."""
    return (round(pose.x + math.cos(pose.theta + math.pi), 6), round(pose.y + math.sin(pose.theta + math.pi), 6))


def _toward(angle: float) -> tuple[float, float]:
    return (round(1.0 + math.cos(YAW + angle), 6), round(2.0 + math.sin(YAW + angle), 6))


@pytest.mark.parametrize(
    ("face", "angle"),
    [("+x", 0.0), ("+y", math.pi / 2), ("-x", math.pi), ("-y", -math.pi / 2), ("xy", math.pi / 4), ("90", -math.pi / 2), (-180, 0.0)],
)
def test_the_queue_forms_at_the_annotated_front(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, face: str | int, angle: float) -> None:
    _, (info,) = _spawn(monkeypatch, tmp_path, {"annotation.yaml": f"{BOX}face: {face!r}\n"})
    assert _queue_side(info.pose) == pytest.approx(_toward(angle), abs=1e-6)
    assert -math.pi <= info.pose.theta <= math.pi


@pytest.mark.parametrize(
    "files",
    [{"annotation.yaml": BOX}, {}, {"annotation.yaml": ""}],
    ids=["annotation without face", "no annotation file", "empty annotation"],
)
def test_an_object_without_an_annotated_face_keeps_its_theta(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, files: dict[str, str | None]) -> None:
    _, (info,) = _spawn(monkeypatch, tmp_path, files)
    assert info.pose.theta == pytest.approx(YAW)
    assert (info.pose.x, info.pose.y) == pytest.approx((1.0, 2.0))


def test_an_unrecognized_face_keeps_the_theta_and_warns_once_per_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sim, objects = _spawn(monkeypatch, tmp_path, {"annotation.yaml": f"{BOX}face: up\n"}, names=("desk_1", "desk_2"))
    assert [o.pose.theta for o in objects] == pytest.approx([YAW, YAW])
    warnings = [line for line in sim._logger.lines if "face" in line]
    assert len(warnings) == 1
