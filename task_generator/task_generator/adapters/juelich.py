"""Juelich pedestrian dynamics experiments (BaSiGo 2013, HERMES 2009) as replay scenarios, one world per distinct lab layout of a set."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path

import attrs
import numpy as np
import shapely
from arena_simulation_setup.generative.world.layout import Area, Layout, compile_layout
from arena_simulation_setup.generative.world.world_generator import save_generated
from arena_simulation_setup.tree.World import World, WorldView
from arena_simulation_setup.tree.World.Scenario import ScenarioView

from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, Attribution, Importer, import_root, write_attribution
from arena_assets.importers.providers import Http, Provider
from task_generator.adapters.eth_ucy import people, scenario_files, tracks

BASE_URL = "https://ped.fz-juelich.de/experiments"
ARCHIVE = "https://ped.fz-juelich.de/da/doku.php?id=start"
LICENSE = "CC-BY-4.0"
METRES = "[in m]"
CM_PER_M = 100.0


@attrs.frozen
class Set:
    """One experiment set of the archive: its data directory, the file prefix of its metadata and trajectory archive, and the frame rate of its frame numbers."""

    directory: str
    prefix: str
    fps: float

    @property
    def metadata(self) -> str:
        return f"{self.directory}/{self.prefix}_metadata.json"

    @property
    def trajectories(self) -> str:
        return f"{self.directory}/{self.prefix}_trajectories_txt.zip"


_BASIGO = "2013.06.19_Duesseldorf_Messe_BaSiGo/data"
_HERMES = "2009.05.12_Duesseldorf_Messe_Hermes/data/zip"

SETS = {
    "basigo_bidirectional": Set(f"{_BASIGO}/2013bidirectional", "2013bidirectional", 25.0),
    "basigo_crossing_90": Set(f"{_BASIGO}/2013crossing_90", "2013crossing_90", 16.0),
    **{f"hermes_{name}": Set(_HERMES, f"2009{name}", 16.0) for name in ("unidirectional_open", "unidirectional_closed", "bidirectional_free", "bidirectional_symmetric", "bidirectional_asymmetric", "bottleneck", "tjunction")},
}


def split_item(item: str) -> tuple[str, str]:
    name, _, run = item.partition("/")
    if name not in SETS or not run:
        raise KeyError(f"no Juelich item '{item}', items are <set>/<run> with the sets {', '.join(SETS)}")
    return name, run


def runs(metadata: dict) -> list[tuple[str, str]]:
    """Name and layout WKT of every run of a set, in archive order."""
    return [(run["run_name"], run["wkt_geometry"]) for run in metadata["experiment"]["run"]]


def layout_index(metadata: dict, run: str) -> int:
    """Index of a run's layout among the distinct layouts of its set, in archive order."""
    layouts = list(dict.fromkeys(wkt for _, wkt in runs(metadata)))
    return layouts.index(dict(runs(metadata))[run])


def world_name(name: str, index: int) -> str:
    return f"juelich_{name}_{index}"


def floor(wkt: str) -> World:
    """A world whose floor is the free area of a layout, its holes walled in."""
    geometry = shapely.from_wkt(wkt)
    return compile_layout(Layout(areas=[Area(part) for part in shapely.get_parts(geometry)]))[0]


def read_run(text: str, fps: float) -> list:
    """Every person of a run file of id, frame, x and y rows, positions in meters whether the file is in centimeters or says it is in meters."""
    scale = 1.0 if METRES in text.partition("\n")[0] else 1.0 / CM_PER_M
    rows = np.loadtxt(io.StringIO(text), comments="#", ndmin=2)
    return people(rows[:, 0].astype(int), rows[:, 1] / fps, rows[:, 2] * scale, rows[:, 3] * scale)


def run_text(archive: Path, run: str) -> str:
    with zipfile.ZipFile(archive) as z:
        member = next((name for name in z.namelist() if Path(name).name == f"{run}.txt"), None)
        if member is None:
            raise KeyError(f"{archive.name} holds no trajectory file {run}.txt")
        return z.read(member).decode("latin-1")


class Juelich(Importer):
    meta = IMPORTERS.meta("scenario", "juelich")

    @classmethod
    def default_provider(cls) -> Provider:
        return Http(BASE_URL)

    @classmethod
    def default_out(cls) -> Path:
        return import_root() / "worlds"

    def metadata(self, name: str) -> dict:
        entry = SETS[name]
        return json.loads(self.provider.get(entry.metadata, self.cache / name / Path(entry.metadata).name).read_text())

    def items(self) -> Iterator[str]:
        for name in SETS:
            for run, _ in runs(self.metadata(name)):
                yield f"{name}/{run}"

    def fetch(self, item: str) -> Path:
        name, run = split_item(item)
        if run not in dict(runs(self.metadata(name))):
            raise KeyError(f"no run '{run}' in the Juelich set {name}")
        entry = SETS[name]
        return self.provider.get(entry.trajectories, self.cache / name / Path(entry.trajectories).name)

    def convert(self, item: str, src: Path, out: Path) -> str:
        name, run = split_item(item)
        metadata = self.metadata(name)
        world = world_name(name, layout_index(metadata, run))
        played = tracks(read_run(run_text(src, run), SETS[name].fps))
        save_generated(WorldView(out / world), floor(dict(runs(metadata))[run]), scenario_files(run, played))
        ScenarioView(out / world / "scenarios" / run).load()
        return f"{world}/scenarios/{run}"

    def import_item(self, item: str, out: Path) -> Path:
        target = super().import_item(item, out)
        write_attribution(target.parents[1] / ATTRIBUTION_FILE, self.attribution(item))
        return target

    def attribution(self, item: str) -> Attribution:
        name, _ = split_item(item)
        experiment = self.metadata(name)["experiment"]
        authors = ", ".join(f"{person['given_name']} {person['family_name']}" for person in experiment.get("persons") or [])
        doi = experiment.get("archive_doi")
        citation = f"{experiment['title']}, Forschungszentrum Juelich pedestrian dynamics data archive" + (f", doi:{doi}" if doi else "")
        return Attribution(source="Juelich pedestrian dynamics data archive", url=f"https://doi.org/{doi}" if doi else ARCHIVE, license=LICENSE, authors=authors or "Forschungszentrum Juelich", citation=citation)
