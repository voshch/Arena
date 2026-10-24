"""SocNavBench curated episodes on the ETH and Zara scenes as replay scenarios with the robot start and goal, in a world per scene whose floor also holds every start and goal."""

from __future__ import annotations

import ast
import configparser
import functools
import math
from collections.abc import Iterable, Iterator
from pathlib import Path

import attrs
import numpy as np
from arena_simulation_setup.generative.world.world_generator import save_generated
from arena_simulation_setup.shared.track import Track
from arena_simulation_setup.tree.World import WorldView
from arena_simulation_setup.tree.World.Scenario import ScenarioView

from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, Attribution, Importer, import_root, write_attribution
from arena_assets.importers.providers import Http, Provider
from task_generator.adapters.eth_ucy import ETH, REPO, REVISION, SEQUENCES, UCY, Sequence, parse_sequence, read_rows, scenario_files, scene_items, tracks, walked_world

BASE_URL = f"https://raw.githubusercontent.com/{REPO}/{REVISION}"
EPISODES_FILE = "params/episode_params_val.ini"
PLACEMENTS_FILE = "params/dataset_params.ini"
DATASETS = "agents/humans/datasets"
TRACK_TIME_STEP_S = 0.001
EPISODE_AUTHORS = "Abhijat Biswas, Allan Wang, Gustavo Silvera, Aaron Steinfeld, Henny Admoni (CMU TBD Lab)"
CITATION = "Biswas, Wang, Silvera, Steinfeld, Admoni, SocNavBench: A Grounded Simulation Testing Framework for Evaluating Social Navigation, ACM THRI 2022"
_DATASET_CITATION = {"eth": ETH, "zara": UCY}


@attrs.frozen
class Placement:
    """How SocNavBench puts one dataset on its map: the file, the rate it is played at and the transform of gather_posn_data."""

    file: str
    fps: float
    offset: tuple[float, float, float]
    swapxy: bool
    flipxn: bool
    flipyn: bool

    @property
    def matrix(self) -> np.ndarray:
        """Linear part taking a raw (x, y) into the map frame."""
        sx, sy = (-1.0 if self.flipxn else 1.0), (-1.0 if self.flipyn else 1.0)
        scaled = np.array([[0.0, sy], [sx, 0.0]]) if self.swapxy else np.array([[sx, 0.0], [0.0, sy]])
        c, s = math.cos(self.offset[2]), math.sin(self.offset[2])
        return np.array([[c, -s], [s, c]]) @ scaled

    @property
    def mirrored(self) -> bool:
        return bool(np.linalg.det(self.matrix) < 0.0)

    def from_map(self, pose: tuple[float, float, float]) -> tuple[float, float, float]:
        """Raw dataset pose of a map pose."""
        inverse = np.linalg.inv(self.matrix)
        x, y = inverse @ (np.array(pose[:2], dtype=float) - self.offset[:2])
        dx, dy = inverse @ np.array([math.cos(pose[2]), math.sin(pose[2])])
        return float(x), float(y), math.atan2(dy, dx)


@attrs.frozen
class Episode:
    """One curated episode as its section states it, start and goal in the map frame."""

    datasets: tuple[str, ...]
    ped_ranges: tuple[tuple[int, int], ...]
    start: tuple[float, float, float]
    goal: tuple[float, float, float]
    max_time: float


def _sections(path: Path, key: str) -> Iterator[configparser.SectionProxy]:
    parser = configparser.ConfigParser(inline_comment_prefixes=(";",), interpolation=None)
    with open(path) as f:
        parser.read_file(f)
    return (parser[name] for name in parser.sections() if key in parser[name])


def _triple(value: Iterable[float]) -> tuple[float, float, float]:
    x, y, z = (float(part) for part in value)
    return x, y, z


def read_placements(path: Path) -> dict[str, Placement]:
    return {
        section.name: Placement(
            file=section["file_name"],
            fps=float(section["fps"]),
            offset=_triple(ast.literal_eval(section["offset"])),
            swapxy=section.getboolean("swapxy", fallback=False),
            flipxn=section.getboolean("flipxn", fallback=False),
            flipyn=section.getboolean("flipyn", fallback=False),
        )
        for section in _sections(path, "file_name")
    }


def read_episodes(path: Path) -> dict[str, Episode]:
    episodes: dict[str, Episode] = {}
    for section in _sections(path, "robot_start_goal"):
        start, goal = (_triple(pose) for pose in ast.literal_eval(section["robot_start_goal"]))
        episodes[section.name] = Episode(
            datasets=tuple(ast.literal_eval(section["pedestrian_datasets"])),
            ped_ranges=tuple((int(first), int(last)) for first, last in ast.literal_eval(section["ped_ranges"])),
            start=start,
            goal=goal,
            max_time=float(section["max_time"]),
        )
    return episodes


def recorded(placement: Placement) -> str | None:
    """Key of the imported sequence a placement plays, None for a file that is not imported."""
    return next((item for item, entry in SEQUENCES.items() if entry.file == placement.file), None)


def left_out(episode: Episode, placements: dict[str, Placement]) -> str:
    """Why an episode is not imported, empty for one that is."""
    reasons: list[str] = []
    if len(episode.datasets) != 1:
        reasons.append(f"it overlays {len(episode.datasets)} datasets")
    for name in episode.datasets:
        placement = placements[name]
        item = recorded(placement)
        if item is None:
            reasons.append(f"its dataset {name} plays {placement.file}, which is not an imported sequence")
            continue
        if placement.fps != SEQUENCES[item].fps:
            reasons.append(f"SocNavBench plays {name} at {placement.fps:g} fps although {item} was recorded at {SEQUENCES[item].fps:g} fps")
        if placement.mirrored:
            reasons.append(f"SocNavBench places {name} as a mirror image of {item}")
    return ", ".join(reasons)


def select(frames: np.ndarray, idents: np.ndarray, ped_range: tuple[int, int], fps: float, max_time: float) -> tuple[int, list[int]]:
    """Frame the episode clock starts at and the pedestrian ids played, as PrerecordedHuman.generate_humans picks them from the file rows."""
    first, last = ped_range
    start_frame = int(frames[0])
    if first == last:
        return start_frame, []
    present = {int(ident) for ident in np.unique(idents)}
    highest = max(present)
    count = -1 if last == -1 else last - first
    if count == -1:
        count = highest - 1
    chosen: list[int] = []
    for i in range(min(count, highest)):
        ident = i + first + 1
        if ident not in present:
            continue
        entered = int(frames[idents == ident][0])
        if i == 0:
            start_frame = entered
        if (entered - start_frame) / fps > max_time:
            break
        chosen.append(ident)
    return start_frame, chosen


def window(track: Track, start: float, end: float) -> Track:
    """The stretch of a track between start and end on a clock counted from start, both cuts interpolated."""
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for index in np.unique(track.segment):
        inside = track.segment == index
        lo, hi = max(float(track.t[inside][0]), start), min(float(track.t[inside][-1]), end)
        if hi - lo < TRACK_TIME_STEP_S:
            continue
        kept = inside & (track.t >= lo + TRACK_TIME_STEP_S) & (track.t <= hi - TRACK_TIME_STEP_S)
        entry, leave = track.state(lo), track.state(hi)
        assert entry is not None and leave is not None
        segments.append(
            (
                np.concatenate(([lo], track.t[kept], [hi])) - start,
                np.concatenate(([entry.x], track.x[kept], [leave.x])),
                np.concatenate(([entry.y], track.y[kept], [leave.y])),
                np.concatenate(([entry.yaw], track.yaw[kept], [leave.yaw])),
            )
        )
    return Track.from_segments(segments)


def scene_pose(placement: Placement, sequence: Sequence, pose: tuple[float, float, float]) -> tuple[float, float, float]:
    """A map pose in the frame of the scene's world."""
    x, y, yaw = placement.from_map(pose)
    return x + sequence.shift[0], y + sequence.shift[1], yaw


def world_name(scene: str) -> str:
    return f"socnavbench_{scene}"


class SocNavBench(Importer):
    meta = IMPORTERS.meta("scenario", "socnavbench")

    @classmethod
    def default_provider(cls) -> Provider:
        return Http(BASE_URL)

    @classmethod
    def default_out(cls) -> Path:
        return import_root() / "worlds"

    @functools.cached_property
    def episodes(self) -> dict[str, Episode]:
        return read_episodes(self.provider.get(EPISODES_FILE, self.cache / EPISODES_FILE))

    @functools.cached_property
    def placements(self) -> dict[str, Placement]:
        return read_placements(self.provider.get(PLACEMENTS_FILE, self.cache / PLACEMENTS_FILE))

    def items(self) -> Iterator[str]:
        return (name for name, episode in self.episodes.items() if not left_out(episode, self.placements))

    def episode(self, item: str) -> Episode:
        """The imported episode of that name."""
        if item not in self.episodes:
            raise KeyError(f"no SocNavBench episode '{item}', items: {', '.join(self.items())}")
        reason = left_out(self.episodes[item], self.placements)
        if reason:
            raise ValueError(f"SocNavBench episode '{item}' is left out: {reason}. Items: {', '.join(self.items())}")
        return self.episodes[item]

    def placement(self, item: str) -> Placement:
        return self.placements[self.episode(item).datasets[0]]

    def sequence(self, item: str) -> Sequence:
        own = recorded(self.placement(item))
        assert own is not None
        return SEQUENCES[own]

    def stops(self, scene: str) -> list[tuple[float, float]]:
        """Start and goal of every imported episode of a scene, in the frame of its world."""
        return [scene_pose(self.placement(item), self.sequence(item), pose)[:2] for item in self.items() if self.sequence(item).scene == scene for pose in (self.episodes[item].start, self.episodes[item].goal)]

    def fetch(self, item: str) -> Path:
        for other in scene_items(self.sequence(item).scene):
            self.provider.get(f"{DATASETS}/{SEQUENCES[other].file}", self.cache / DATASETS / SEQUENCES[other].file)
        return self.cache

    def convert(self, item: str, src: Path, out: Path) -> str:
        episode, placement, sequence = self.episode(item), self.placement(item), self.sequence(item)
        own = recorded(placement)
        assert own is not None
        bodies = {other: parse_sequence(src / DATASETS / SEQUENCES[other].file, SEQUENCES[other].fps, SEQUENCES[other].shift) for other in scene_items(sequence.scene)}
        frames, idents, _, _ = read_rows(src / DATASETS / sequence.file)
        start_frame, chosen = select(frames.astype(int), idents.astype(int), episode.ped_ranges[0], placement.fps, episode.max_time)
        start = (start_frame - float(frames.min())) / sequence.fps
        names = {f"ped_{ident}" for ident in chosen}
        cut = {name: window(track, start, start + episode.max_time) for name, track in tracks([body for body in bodies[own] if body.name in names]).items()}
        robot = {
            "start": [round(value, 3) for value in scene_pose(placement, sequence, episode.start)],
            "phases": [{"goto": [round(value, 3) for value in scene_pose(placement, sequence, episode.goal)]}],
        }
        world = world_name(sequence.scene)
        floor = walked_world([body for other in bodies.values() for body in other], self.stops(sequence.scene))
        files = scenario_files(item, {name: track for name, track in cut.items() if len(track.t)}, [robot])
        save_generated(WorldView(out / world), floor, files)
        ScenarioView(out / world / "scenarios" / item).load()
        return f"{world}/scenarios/{item}"

    def import_item(self, item: str, out: Path) -> Path:
        target = super().import_item(item, out)
        write_attribution(target.parents[1] / ATTRIBUTION_FILE, self.attribution(item))
        return target

    def attribution(self, item: str) -> Attribution:
        sequence = self.sequence(item)
        return Attribution(
            source="SocNavBench episodes on the ETH and UCY pedestrian datasets",
            url=f"https://github.com/{REPO}/blob/{REVISION}/{EPISODES_FILE}",
            license="MIT for the episodes, unspecified for the trajectories",
            authors=f"{sequence.authors}, episodes by {EPISODE_AUTHORS}",
            citation=f"{_DATASET_CITATION[sequence.scene]}. {CITATION}",
        )
