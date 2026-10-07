"""ETH and UCY pedestrian sequences (SocNavBench's interpolated copies) as replay scenarios, in a world per scene whose floor is the area its pedestrians walked."""

from __future__ import annotations

import csv
import itertools
import math
from collections.abc import Iterable, Iterator
from pathlib import Path

import attrs
import numpy as np
import shapely
import yaml
from arena_simulation_setup.generative.world.layout import Area, Layout, compile_layout
from arena_simulation_setup.generative.world.world_generator import save_generated
from arena_simulation_setup.shared.track import Track
from arena_simulation_setup.tree.World import World, WorldView
from arena_simulation_setup.tree.World.Scenario import SCENARIO_FORMAT, ScenarioView

from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, Attribution, Importer, import_root, write_attribution
from arena_assets.importers.providers import Http, Provider

REPO = "CMU-TBD/SocNavBench"
REVISION = "2724ea85ff22ca61a88ee95285dfc3fa056656c6"
BASE_URL = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/agents/humans/datasets"
TRACK_HZ = 10.0
MIN_HEADING_SPEED = 0.1
HEADING_WINDOW_S = 0.2
MAX_GAP_S = 1.0
WORLD_RESOLUTION_M = 0.05
ROUTE_TOLERANCE_M = 0.25
TIME_PACE = 1.0
MIN_MOVING_SPEED = 0.1
MIN_STAY_S = 1.0
ENTRY_WINDOW_S = 0.4
JUMP_M = 0.5
JUMP_SPEED = 6.0
MODEL = "arenian"
AGENT_TYPE = "adult"
FOOTPRINT_M = 0.5
HULL_RATIO = 0.25
STRAIGHTEN_M = 0.15
ETH = "Pellegrini, Ess, Schindler, Van Gool, You'll Never Walk Alone: Modeling Social Behavior for Multi-target Tracking, ICCV 2009"
UCY = "Lerner, Chrysanthou, Lischinski, Crowds by Example, Computer Graphics Forum 26(3), 2007"
SOCNAVBENCH = "Biswas, Wang, Silvera, Steinfeld, Admoni, SocNavBench: A Grounded Simulation Testing Framework for Evaluating Social Navigation, ACM THRI 2022"


@attrs.frozen
class Sequence:
    """One recorded sequence: its scene, file, the video frame rate its frame numbers count and the shift of its frame into the scene's."""

    scene: str
    file: str
    fps: float
    authors: str
    citation: str
    shift: tuple[float, float] = (0.0, 0.0)

    @property
    def world(self) -> str:
        return f"eth_ucy_{self.scene}"


_ETH_AUTHORS = "Stefano Pellegrini, Andreas Ess, Konrad Schindler, Luc Van Gool (ETH Zurich, BIWI), converted by the SocNavBench team (CMU TBD Lab)"
_UCY_AUTHORS = "Alon Lerner, Yiorgos Chrysanthou, Dani Lischinski (University of Cyprus), converted by the SocNavBench team (CMU TBD Lab)"

SEQUENCES = {
    "eth": Sequence("eth", "eth/world_coordinate_inter_eth.csv", 15.0, _ETH_AUTHORS, f"{ETH}. {SOCNAVBENCH}"),
    "hotel": Sequence("hotel", "hotel/world_coordinate_inter_hotel.csv", 25.0, _ETH_AUTHORS, f"{ETH}. {SOCNAVBENCH}"),
    "univ": Sequence("univ", "univ/world_coordinate_inter.csv", 25.0, _UCY_AUTHORS, f"{UCY}. {SOCNAVBENCH}"),
    "zara01": Sequence("zara", "zara01/world_coordinate_inter.csv", 25.0, _UCY_AUTHORS, f"{UCY}. {SOCNAVBENCH}"),
    "zara02": Sequence("zara", "zara02/world_coordinate_inter.csv", 25.0, _UCY_AUTHORS, f"{UCY}. {SOCNAVBENCH}", shift=(0.0, 1.15)),
}


def sequence(item: str) -> Sequence:
    try:
        return SEQUENCES[item]
    except KeyError:
        raise KeyError(f"no ETH/UCY sequence '{item}', items: {', '.join(SEQUENCES)}") from None


def scene_items(scene: str) -> list[str]:
    return [item for item, entry in SEQUENCES.items() if entry.scene == scene]


@attrs.frozen
class Body:
    name: str
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray


def heading_velocity(x: np.ndarray, y: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Velocity over a HEADING_WINDOW_S span centered on each sample, one-sided at the ends."""
    k = max(1, round(rate * HEADING_WINDOW_S / 2.0))
    i = np.arange(len(x))
    lo = np.clip(i - k, 0, len(x) - 1)
    hi = np.clip(i + k, 0, len(x) - 1)
    dt = (hi - lo) / rate
    return (x[hi] - x[lo]) / dt, (y[hi] - y[lo]) / dt


def resample(body: Body, t0: float, rate: float, max_gap: float, min_heading_speed: float) -> Track:
    """Track on the rate grid counted from t0, gaps up to max_gap bridged and longer ones left open, heading along the motion and held while standing."""
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    breaks = np.flatnonzero(np.diff(body.t) > max_gap) + 1
    last_yaw = 0.0
    for seg_t, seg_x, seg_y in zip(np.split(body.t, breaks), np.split(body.x, breaks), np.split(body.y, breaks), strict=True):
        first = int(np.ceil((seg_t[0] - t0) * rate - 1e-9))
        last = int(np.floor((seg_t[-1] - t0) * rate + 1e-9))
        if last - first < 1:
            continue
        grid = np.arange(first, last + 1) / rate
        x = np.interp(grid + t0, seg_t, seg_x)
        y = np.interp(grid + t0, seg_t, seg_y)
        vx, vy = heading_velocity(x, y, rate)
        moving = np.hypot(vx, vy) >= min_heading_speed
        if moving.any():
            filled = np.flatnonzero(moving)
            source = filled[np.maximum(np.searchsorted(filled, np.arange(len(grid)), "right") - 1, 0)]
            yaw = np.arctan2(vy, vx)[source]
        else:
            yaw = np.full(len(grid), last_yaw)
        last_yaw = float(yaw[-1])
        segments.append((grid, x, y, yaw))
    return Track.from_segments(segments)


def route(track: Track) -> list[tuple[float, float, float]]:
    """Corners of the walked path past its start, each with the heading it is reached at."""
    if len(track.t) < 2:
        return []
    line = shapely.LineString(np.column_stack([track.x, track.y])).simplify(ROUTE_TOLERANCE_M, preserve_topology=False)
    if line.length < ROUTE_TOLERANCE_M:
        return []
    corners = list(line.coords)
    return [(round(bx, 3), round(by, 3), round(math.atan2(by - ay, bx - ax), 3)) for (ax, ay), (bx, by) in zip(corners, corners[1:], strict=False)]


def walking_speed(track: Track) -> float:
    speeds = (np.hypot(np.diff(track.x), np.diff(track.y)) / np.diff(track.t))[np.diff(track.segment) == 0]
    moving = speeds[speeds >= MIN_HEADING_SPEED]
    return round(float(np.median(moving)), 2) if len(moving) else 0.0


def _pose(track: Track) -> list[float]:
    return [round(float(track.x[0]), 3), round(float(track.y[0]), 3), round(float(track.yaw[0]), 3)]


def simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Indices of the Douglas-Peucker corners of a polyline in any dimension."""
    keep = np.zeros(len(points), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        chord = points[b] - points[a]
        length = float(np.linalg.norm(chord))
        rel = points[a + 1 : b] - points[a]
        off = rel - np.outer(rel @ chord, chord) / length**2 if length > 1e-9 else rel
        distance = np.linalg.norm(off, axis=1)
        far = int(np.argmax(distance))
        if distance[far] > tolerance:
            keep[a + 1 + far] = True
            stack += [(a, a + 1 + far), (a + 1 + far, b)]
    return np.flatnonzero(keep)


def timed_legs(t: np.ndarray, xy: np.ndarray) -> list[list[float]]:
    """Legs [x, y, speed, dwell] of a stay within ROUTE_TOLERANCE_M of it in space and, at TIME_PACE, in time, each stop folded into the dwell of the leg before."""
    corners = simplify(np.column_stack([xy, TIME_PACE * (t - t[0])]), ROUTE_TOLERANCE_M)
    legs: list[list[float]] = []
    for i, j in itertools.pairwise(corners):
        length = float(np.hypot(*(xy[j] - xy[i])))
        span = float(t[j] - t[i])
        if length / span >= MIN_MOVING_SPEED:
            legs.append([float(xy[j, 0]), float(xy[j, 1]), length / span, 0.0])
        elif legs:
            legs[-1][3] += span
        else:
            legs.append([float(xy[0, 0]), float(xy[0, 1]), MIN_MOVING_SPEED, span])
    return [[round(x, 3), round(y, 3), round(speed, 3), round(dwell, 2)] for x, y, speed, dwell in legs]


def stays(track: Track) -> list[dict]:
    """Each continuous stay of a track, split at its gaps and at jumps farther than JUMP_M faster than JUMP_SPEED, with its start, entry pose and velocity and timed legs."""
    xy = np.column_stack([track.x, track.y])
    step = np.hypot(*np.diff(xy, axis=0).T)
    jump = (step > JUMP_M) & (step > JUMP_SPEED * np.diff(track.t))
    index = track.segment + np.concatenate(([0], np.cumsum(jump)))
    result = []
    for k in np.unique(index):
        t, part = track.t[index == k], xy[index == k]
        if len(t) < 2 or t[-1] - t[0] < MIN_STAY_S:
            continue
        ahead = max(int(np.searchsorted(t, t[0] + ENTRY_WINDOW_S, "right")) - 1, 1)
        vx, vy = (part[ahead] - part[0]) / (t[ahead] - t[0])
        corners = simplify(part, ROUTE_TOLERANCE_M)
        walked = float(np.hypot(*np.diff(part[corners], axis=0).T).sum()) >= ROUTE_TOLERANCE_M
        if math.hypot(vx, vy) >= MIN_MOVING_SPEED:
            heading = math.atan2(vy, vx)
        elif walked:
            heading = math.atan2(part[corners[1], 1] - part[0, 1], part[corners[1], 0] - part[0, 0])
        else:
            heading = 0.0
        result.append(
            {
                "start": round(float(t[0]), 3),
                "pose": [round(float(part[0, 0]), 3), round(float(part[0, 1]), 3), round(heading, 3)],
                "velocity": [round(float(vx), 3), round(float(vy), 3)],
                "legs": timed_legs(t, part),
            }
        )
    return result


def pedestrian(name: str, track: Track) -> dict:
    speed = walking_speed(track)
    return {
        "name": name,
        "model": MODEL,
        "pose": _pose(track),
        "velocity": speed,
        "agent": {"agent_type": AGENT_TYPE, "desired_velocity": speed},
        "waypoints": [[x, y, 0.0] for x, y, _ in route(track)],
        "waypoint_mode": "once",
        "stays": stays(track),
        "track": f"./{name}.csv",
    }


def read_rows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The rows frame, id, y, x of a transposed sequence file, one column per sample in file order."""
    with open(path, newline="") as f:
        rows = [np.array(row, dtype=float) for row in csv.reader(f)]
    if len(rows) != 4:
        raise ValueError(f"{path}: {len(rows)} rows, expected the four rows frame, id, y, x")
    return rows[0], rows[1], rows[2], rows[3]


def people(ident: np.ndarray, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> list[Body]:
    """A body per id of parallel id, time, x and y columns, samples sorted by time with repeated times dropped, ids seen at fewer than two times left out."""
    bodies: list[Body] = []
    for value in np.unique(ident):
        mask = ident == value
        order = np.argsort(t[mask], kind="stable")
        times, keep = np.unique(t[mask][order], return_index=True)
        if len(times) < 2:
            continue
        bodies.append(Body(f"ped_{int(value)}", times, x[mask][order][keep], y[mask][order][keep]))
    return bodies


def parse_sequence(path: Path, fps: float, shift: tuple[float, float] = (0.0, 0.0)) -> list[Body]:
    """Every pedestrian of a transposed frame, id, y, x file, timed in seconds from the sequence's first frame and moved by *shift*."""
    frame, ident, y, x = read_rows(path)
    return people(ident, (frame - frame.min()) / fps, x + shift[0], y + shift[1])


def walked_area(bodies: list[Body], stops: Iterable[tuple[float, float]] = ()) -> shapely.Polygon:
    """Floor of a scene: the concave hull of every walked path and stop, widened by a footprint and straightened."""
    if not bodies:
        raise ValueError("no pedestrian walked the scene")
    paths = [shapely.LineString(np.column_stack([body.x, body.y])) for body in bodies]
    marks = np.array(list(stops), dtype=float).reshape(-1, 2)
    hull = shapely.concave_hull(shapely.MultiPoint(np.concatenate([*(path.coords for path in paths), marks])), ratio=HULL_RATIO)
    floor = shapely.union_all([hull, *paths, *(shapely.Point(mark) for mark in marks)]).buffer(FOOTPRINT_M + STRAIGHTEN_M).simplify(STRAIGHTEN_M)
    if not isinstance(floor, shapely.Polygon):
        raise ValueError(f"the walked floor came out as a {floor.geom_type}, not one polygon")
    return shapely.Polygon(floor.exterior)


def walked_world(bodies: list[Body], stops: Iterable[tuple[float, float]] = ()) -> World:
    return compile_layout(Layout(areas=[Area(walked_area(bodies, stops))]))[0]


def tracks(bodies: list[Body]) -> dict[str, Track]:
    """Track of every pedestrian on the sequence clock, those too short to resample left out."""
    resampled = {body.name: resample(body, 0.0, TRACK_HZ, MAX_GAP_S, MIN_HEADING_SPEED) for body in bodies}
    return {name: track for name, track in resampled.items() if len(track.t)}


def scenario_files(item: str, played: dict[str, Track], robots: Iterable[dict] = ()) -> dict[str, bytes]:
    """World-relative files of one scenario: scenario.yaml and a track per pedestrian."""
    base = f"scenarios/{item}"
    files = {f"{base}/{name}.csv": track.dumps() for name, track in played.items()}
    scenario = {"format": SCENARIO_FORMAT, "humans": [pedestrian(name, track) for name, track in played.items()], "robots": list(robots)}
    files[f"{base}/scenario.yaml"] = yaml.safe_dump(scenario, sort_keys=False, default_flow_style=None, width=120).encode()
    return files


class ETHUCY(Importer):
    meta = IMPORTERS.meta("scenario", "eth_ucy")

    @classmethod
    def default_provider(cls) -> Provider:
        return Http(BASE_URL)

    @classmethod
    def default_out(cls) -> Path:
        return import_root() / "worlds"

    def items(self) -> Iterator[str]:
        return iter(SEQUENCES)

    def fetch(self, item: str) -> Path:
        wanted = sequence(item)
        for other in scene_items(wanted.scene):
            entry = SEQUENCES[other]
            self.provider.get(entry.file, self.cache / entry.file)
        return self.cache / wanted.file

    def convert(self, item: str, src: Path, out: Path) -> str:
        wanted = sequence(item)
        scene = [body for other in scene_items(wanted.scene) for body in parse_sequence(src.parents[1] / SEQUENCES[other].file, SEQUENCES[other].fps, SEQUENCES[other].shift)]
        target = out / wanted.world
        save_generated(WorldView(target), walked_world(scene), scenario_files(item, tracks(parse_sequence(src, wanted.fps, wanted.shift))))
        ScenarioView(target / "scenarios" / item).load()
        return f"{wanted.world}/scenarios/{item}"

    def import_item(self, item: str, out: Path) -> Path:
        target = super().import_item(item, out)
        write_attribution(out / sequence(item).world / ATTRIBUTION_FILE, self.attribution(item))
        return target

    def attribution(self, item: str) -> Attribution:
        wanted = sequence(item)
        return Attribution(source="ETH and UCY pedestrian datasets via SocNavBench", url=f"https://github.com/{REPO}/blob/{REVISION}/agents/humans/datasets/{wanted.file}", license="unspecified", authors=wanted.authors, citation=wanted.citation)
