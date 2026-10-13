"""Eindhoven GLOW 2019 crowd recording (Corbetta et al.) as replay scenarios of 5-minute periods on an open floor around the tracked area."""

from __future__ import annotations

import datetime
import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from arena_simulation_setup.generative.world.world_generator import save_generated
from arena_simulation_setup.tree.World import WorldView
from arena_simulation_setup.tree.World.Scenario import ScenarioView

from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, Attribution, Importer, import_root, write_attribution
from arena_assets.importers.providers import Http, Provider
from task_generator.adapters.eth_ucy import open_floor, people, scenario_files, tracks

RECORD = 7007358
BASE_URL = f"https://zenodo.org/records/{RECORD}/files"
RECORDING = "glow19_pnasnexus.csv"
INDEX = "glow19_periods.json"
HEADER = "time,tracked_object,x,y,x_sav,y_sav,vx_sav,vy_sav"
PERIOD_S = 300.0
MARGIN_M = 2.0
WORLD = "glow"
LICENSE = "CC-BY-4.0"
AUTHORS = "Alessandro Gabbana, Federico Toschi, Philip Ross, Antal Haans, Alessandro Corbetta"
CITATION = "Gabbana, Toschi, Ross, Haans, Corbetta, Fluctuations in pedestrian dynamics routing choices, PNAS Nexus 2022, doi:10.5281/zenodo.7007358"
ITEM_FORMAT = "%Y-%m-%dT%H%M"

_EPOCH = datetime.datetime(1970, 1, 1)


def seconds(stamp: str) -> float:
    """Seconds of a recorded wall-clock time since the epoch, in the recording's own time zone."""
    return (datetime.datetime.fromisoformat(stamp) - _EPOCH).total_seconds()


def period_item(period: int) -> str:
    return (_EPOCH + datetime.timedelta(seconds=period * PERIOD_S)).strftime(ITEM_FORMAT)


def item_start(item: str) -> float:
    return (datetime.datetime.strptime(item, ITEM_FORMAT) - _EPOCH).total_seconds()


def index(recording: Path) -> dict:
    """Byte span of every period of the time-sorted recording and the bounds of every smoothed position."""
    spans: dict[str, list[int]] = {}
    min_x = min_y = math.inf
    max_x = max_y = -math.inf
    current = None
    span: list[int] = []
    with open(recording, "rb") as f:
        header = f.readline().decode().strip()
        if header != HEADER:
            raise ValueError(f"{recording.name}: header '{header}', expected '{HEADER}'")
        offset = f.tell()
        for line in f:
            stamp, _, _, _, x, y, *_ = line.decode().split(",")
            period = int(seconds(stamp) // PERIOD_S)
            if period != current:
                if current is not None and period < current:
                    raise ValueError(f"{recording.name}: rows are not sorted by time at byte {offset}")
                span = spans[period_item(period)] = [offset, offset]
                current = period
            offset += len(line)
            span[1] = offset
            px, py = float(x), float(y)
            min_x, max_x = min(min_x, px), max(max_x, px)
            min_y, max_y = min(min_y, py), max(max_y, py)
    return {"periods": spans, "bounds": [min_x, min_y, max_x, max_y]}


def read_period(recording: Path, span: list[int], start: float) -> list:
    """Every person of one period, timed from the period start, at its smoothed positions."""
    with open(recording, "rb") as f:
        f.seek(span[0])
        lines = f.read(span[1] - span[0]).decode().splitlines()
    fields = [line.split(",") for line in lines if line]
    ident = np.array([int(row[1]) for row in fields])
    t = np.array([seconds(row[0]) - start for row in fields])
    x = np.array([float(row[4]) for row in fields])
    y = np.array([float(row[5]) for row in fields])
    return people(ident, t, x, y)


class GLOW(Importer):
    meta = IMPORTERS.meta("scenario", "glow")

    @classmethod
    def default_provider(cls) -> Provider:
        return Http(BASE_URL)

    @classmethod
    def default_out(cls) -> Path:
        return import_root() / "worlds"

    def recording(self) -> Path:
        return self.provider.get(RECORDING, self.cache / RECORDING)

    def periods(self) -> dict:
        path = self.cache / INDEX
        if not path.is_file():
            built = index(self.recording())
            partial = path.with_name(path.name + ".part")
            partial.write_text(json.dumps(built))
            partial.replace(path)
        return json.loads(path.read_text())

    def items(self) -> Iterator[str]:
        return iter(self.periods()["periods"])

    def fetch(self, item: str) -> Path:
        if item not in self.periods()["periods"]:
            raise KeyError(f"no GLOW period '{item}', items are the 5-minute periods of the recording as {ITEM_FORMAT.replace('%', '')}")
        return self.recording()

    def convert(self, item: str, src: Path, out: Path) -> str:
        known = self.periods()
        low, high = known["bounds"][:2], known["bounds"][2:]
        floor = open_floor(np.array([low, high]), MARGIN_M)
        played = tracks(read_period(src, known["periods"][item], item_start(item)))
        save_generated(WorldView(out / WORLD), floor, scenario_files(item, played))
        ScenarioView(out / WORLD / "scenarios" / item).load()
        return f"{WORLD}/scenarios/{item}"

    def import_item(self, item: str, out: Path) -> Path:
        target = super().import_item(item, out)
        write_attribution(target.parents[1] / ATTRIBUTION_FILE, self.attribution(item))
        return target

    def attribution(self, item: str) -> Attribution:
        return Attribution(source="Eindhoven GLOW 2019 crowd recording", url=f"https://zenodo.org/records/{RECORD}", license=LICENSE, authors=AUTHORS, citation=CITATION)
