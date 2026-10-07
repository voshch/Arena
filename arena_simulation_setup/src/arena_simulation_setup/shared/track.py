"""Timed ground track of one pedestrian, the file a scenario human names under track, and the timed legs of its stays."""

from __future__ import annotations

import io
import math
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path

import attrs
import numpy as np

from arena_simulation_setup.utils.cattrs import Parseable
from arena_simulation_setup.utils.geometry import Pose, Position

if typing.TYPE_CHECKING:
    from arena_simulation_setup.shared.entities import DynamicObstacle

HEADER = "t,x,y,yaw"
_FORMAT = ("%.3f", "%.4f", "%.4f", "%.4f")


@attrs.frozen
class TrackState:
    x: float
    y: float
    yaw: float
    vx: float
    vy: float


@attrs.frozen(eq=False)
class Track:
    """Rows of t (s from episode start), x, y (m, level frame) and yaw (rad), split into segments the pedestrian is absent between."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    segment: np.ndarray

    @classmethod
    def from_segments(cls, segments: Sequence[Sequence[np.ndarray]]) -> Track:
        kept = [[np.asarray(column, dtype=float) for column in rows] for rows in segments if len(rows[0])]
        columns = [np.concatenate([rows[i] for rows in kept]) if kept else np.zeros(0) for i in range(4)]
        index = np.concatenate([np.full(len(rows[0]), n, dtype=np.int64) for n, rows in enumerate(kept)]) if kept else np.zeros(0, dtype=np.int64)
        if np.any(np.diff(columns[0]) <= 0.0):
            raise ValueError("track times must increase strictly")
        t, x, y, yaw = columns
        return cls(t, x, y, yaw, index)

    @classmethod
    def load(cls, path: Path) -> Track:
        with open(path) as f:
            header = f.readline().strip()
            if header != HEADER:
                raise ValueError(f"{path}: header '{header}', expected '{HEADER}'")
            rows = np.loadtxt(f, delimiter=",", ndmin=2)
        if rows.shape[1] != 4:
            raise ValueError(f"{path}: {rows.shape[1]} columns, expected '{HEADER}'")
        lifted = np.isnan(rows).any(axis=1)
        index = np.cumsum(lifted)[~lifted]
        rows = rows[~lifted]
        if np.any(np.diff(rows[:, 0]) <= 0.0):
            raise ValueError(f"{path}: track times must increase strictly")
        return cls(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], np.unique(index, return_inverse=True)[1])

    def dumps(self) -> bytes:
        rows = np.column_stack([self.t, self.x, self.y, self.yaw])
        starts = np.flatnonzero(np.diff(self.segment)) + 1
        buffer = io.BytesIO()
        np.savetxt(buffer, np.insert(rows, starts, np.nan, axis=0), fmt=_FORMAT, delimiter=",", header=HEADER, comments="")
        return buffer.getvalue()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.dumps())

    @property
    def start(self) -> float:
        return float(self.t[0])

    @property
    def end(self) -> float:
        return float(self.t[-1])

    @property
    def spans(self) -> list[tuple[float, float]]:
        starts = np.concatenate(([0], np.flatnonzero(np.diff(self.segment)) + 1))
        ends = np.concatenate((starts[1:], [len(self.t)])) - 1
        return [(float(self.t[a]), float(self.t[b])) for a, b in zip(starts, ends, strict=True)]

    def shifted(self, dx: float, dy: float) -> Track:
        return attrs.evolve(self, x=self.x + dx, y=self.y + dy)

    def state(self, t: float, hold_start: bool = False) -> TrackState | None:
        """Interpolated pose and velocity at t, None while the pedestrian is absent, at rest on the first sample before it with hold_start."""
        i = int(np.searchsorted(self.t, t, "right")) - 1
        if i < 0:
            return TrackState(float(self.x[0]), float(self.y[0]), float(self.yaw[0]), 0.0, 0.0) if hold_start else None
        last = i + 1 >= len(self.t) or self.segment[i + 1] != self.segment[i]
        if last and t > self.t[i]:
            return None
        a = i - 1 if last else i
        if a < 0 or self.segment[a] != self.segment[a + 1]:
            return TrackState(float(self.x[i]), float(self.y[i]), float(self.yaw[i]), 0.0, 0.0)
        dt = float(self.t[a + 1] - self.t[a])
        u = (t - float(self.t[a])) / dt
        turn = (float(self.yaw[a + 1] - self.yaw[a]) + math.pi) % (2.0 * math.pi) - math.pi
        return TrackState(
            x=float(self.x[a] + u * (self.x[a + 1] - self.x[a])),
            y=float(self.y[a] + u * (self.y[a + 1] - self.y[a])),
            yaw=(float(self.yaw[a]) + u * turn + math.pi) % (2.0 * math.pi) - math.pi,
            vx=float(self.x[a + 1] - self.x[a]) / dt,
            vy=float(self.y[a + 1] - self.y[a]) / dt,
        )


def track_file(entry: DynamicObstacle) -> Path | None:
    """Track file of a scenario human, a relative path resolves against the scenario directory."""
    if not entry.track:
        return None
    path = Path(entry.track)
    if path.is_absolute() or entry.included_from is None:
        return path
    return Path(entry.included_from) / path


@attrs.frozen
class Leg(Parseable):
    """Walk to a point at a speed in m/s, then stand there for dwell seconds."""

    position: Position
    speed: float
    dwell: float = 0.0

    @classmethod
    def parse(cls, value: Leg | Sequence[float]) -> Leg:
        if isinstance(value, Leg):
            return value
        x, y, speed, dwell = (float(item) for item in value)
        return cls(Position.parse([x, y]), speed, dwell)


@attrs.frozen
class Stay(Parseable):
    """One continuous stay of a recorded pedestrian: its start on the episode clock, entry pose and velocity, and the legs it walks."""

    start: float
    pose: Pose
    velocity: tuple[float, float]
    legs: tuple[Leg, ...]

    @classmethod
    def parse(cls, value: Stay | Mapping) -> Stay:
        if isinstance(value, Stay):
            return value
        unknown = set(value) - {"start", "pose", "velocity", "legs"}
        if unknown:
            raise ValueError(f"stay has unknown keys {sorted(unknown)}, expected start, pose, velocity and legs")
        vx, vy = (float(item) for item in value["velocity"])
        return cls(float(value["start"]), Pose.parse(value["pose"]), (vx, vy), tuple(Leg.parse(leg) for leg in value["legs"]))


def parse_stays(value: object) -> list[Stay]:
    return [Stay.parse(item) for item in typing.cast("Sequence", value or [])]
