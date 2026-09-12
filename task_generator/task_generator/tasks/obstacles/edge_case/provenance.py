"""Per-episode case records.

The scorer joins trials to what was built on this file, not on directory names: the record
carries every effect as designed, the solved interception, and the outcome of the build.
Written append-only as JSON Lines so a run that dies halfway still leaves usable records.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs


@attrs.frozen
class CaseRecord:
    """One episode's worth of edge-case provenance."""

    run_seed: str
    episode_id: int
    world: str
    seed: int

    scenario: str = ""
    case_id: str = ""
    base: str = ""
    prompt_id: str = ""
    steps: tuple[str, ...] = ()

    #: Every effect the block declared, as designed.
    effects: tuple[Mapping[str, Any], ...] = ()
    #: Every route-rewrite plan armed, resolved (targets, owned agents).
    plans: tuple[Mapping[str, Any], ...] = ()

    # --- injected pedestrians -----------------------------------------------------------
    injected: int = 0
    #: First injected agent, whose geometry is the one recorded below.
    target_agent: str = ""
    inject_type: str = ""
    inject_model: str = ""
    waypoint_mode: str = ""
    #: "intercept" (a walk into the encounter) | "stand" (placed at the encounter) | ""
    place: str = ""
    after: str = ""
    ped_speed: float | None = None

    # --- the encounter, as designed and as achieved --------------------------------------
    approach_angle_requested: float | None = None
    approach_angle_achieved: float | None = None
    encounter_at_requested: float | None = None
    encounter_at_achieved: float | None = None
    encounter_point: tuple[float, float] | None = None
    t_encounter: float | None = None
    designed_ttc: float | None = None
    designed_pet: float | None = None
    co_arrival_offset: float | None = None
    lead_out_achieved: float | None = None
    spawn_pose: tuple[float, float] | None = None
    ped_goal: tuple[float, float] | None = None
    robot_leg: tuple[tuple[float, float], tuple[float, float]] | None = None
    robot_route: tuple[tuple[float, float], ...] | None = None
    route_length: float | None = None
    route_planned: bool | None = None
    geometry_retries: int | None = None
    #: Sim seconds the robot mode held the robot at its start; the interception waited the same.
    robot_hold: float = 0.0
    #: Seconds the interception's walker waits at its spawn (a lead-in that did not fit in free space).
    depart_at: float = 0.0
    traversals_seen: int | None = None

    #: Base-scenario agents whose spawn failed the free-space check. Non-fatal but recorded.
    invalid_spawns: tuple[str, ...] = ()

    # --- object timeline, as designed ----------------------------------------------------
    object_timeline: str = ""
    object_events: tuple[Mapping[str, Any], ...] = ()

    status: str = "ok"
    reason: str = ""

    created_at: float = attrs.Factory(time.time)

    def to_json(self) -> str:
        return json.dumps(attrs.asdict(self), separators=(",", ":"), sort_keys=True, default=str)


def default_record_dir() -> Path:
    """``$ARENA_DATA_DIR/edge_case`` when set, else a temp dir."""
    data_dir = os.environ.get("ARENA_DATA_DIR")
    if data_dir:
        return Path(data_dir) / "edge_case"
    return Path(tempfile.gettempdir()) / "arena_edge_case"


class CaseLog:
    """Append-only writer for ``cases.jsonl``."""

    def __init__(self, record_dir: Path | str | None = None) -> None:
        self._dir = Path(record_dir) if record_dir else default_record_dir()
        self._path = self._dir / "cases.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: CaseRecord) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as fh:
            fh.write(record.to_json() + "\n")
        return self._path

    def read_all(self) -> Sequence[dict[str, Any]]:
        if not self._path.is_file():
            return []
        return [json.loads(line) for line in self._path.read_text().splitlines() if line.strip()]
