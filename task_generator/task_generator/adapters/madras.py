"""MADRAS TopView recordings of the Fete des Lumieres 2022 in Lyon as replay scenarios on an open floor around the tracked area."""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from arena_simulation_setup.generative.world.world_generator import save_generated
from arena_simulation_setup.tree.World import WorldView
from arena_simulation_setup.tree.World.Scenario import ScenarioView

from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, Attribution, Importer, import_root, write_attribution
from arena_assets.importers.providers import Http, Provider
from task_generator.adapters.eth_ucy import open_floor, people, scenario_files, tracks

RECORD = 13830435
BASE_URL = f"https://zenodo.org/records/{RECORD}/files"
ARCHIVE = "Data_Madras.zip"
FPS = 30.0
MARGIN_M = 2.0
WORLD = "madras_topview"
LICENSE = "CC-BY-4.0"
AUTHORS = "Oscar Dufour, Huu-Tu Dang, Jakob Cordes, Raphael Korbmacher, Benoit Gaudou, Mohcine Chraibi, Alexandre Nicolas, Antoine Tordeux"
CITATION = "Dufour et al., Dense Crowd Dynamics and Pedestrian Trajectories: A Multiscale Field Study at the Fete des Lumieres in Lyon, doi:10.5281/zenodo.13830435"

_SEQUENCE = re.compile(r"(?:^|/)TopView_trajectories/TopView_(?P<name>\w+)\.txt$")


def sequences(archive: Path) -> dict[str, str]:
    """Item name and archive member of every TopView sequence, in name order."""
    with zipfile.ZipFile(archive) as z:
        found = {f"topview_{m['name'].lower()}": name for name in z.namelist() if (m := _SEQUENCE.search(name))}
    return dict(sorted(found.items()))


def read_sequence(text: str) -> tuple[np.ndarray, list]:
    """Every tracked point and person of a sequence of id, frame, x and y rows in the local frame in meters, timed from frame 0."""
    rows = np.loadtxt(io.StringIO(text), comments="#", ndmin=2)
    return rows[:, 2:4], people(rows[:, 0].astype(int), rows[:, 1] / FPS, rows[:, 2], rows[:, 3])


class MADRAS(Importer):
    meta = IMPORTERS.meta("scenario", "madras")

    @classmethod
    def default_provider(cls) -> Provider:
        return Http(BASE_URL)

    @classmethod
    def default_out(cls) -> Path:
        return import_root() / "worlds"

    def items(self) -> Iterator[str]:
        return iter(sequences(self.fetch_archive()))

    def fetch_archive(self) -> Path:
        return self.provider.get(ARCHIVE, self.cache / ARCHIVE)

    def fetch(self, item: str) -> Path:
        archive = self.fetch_archive()
        if item not in sequences(archive):
            raise KeyError(f"no MADRAS sequence '{item}', items: {', '.join(sequences(archive))}")
        return archive

    def convert(self, item: str, src: Path, out: Path) -> str:
        members = sequences(src)
        with zipfile.ZipFile(src) as z:
            read = {name: read_sequence(z.read(member).decode("latin-1")) for name, member in members.items()}
        floor = open_floor(np.vstack([points for points, _ in read.values()]), MARGIN_M)
        save_generated(WorldView(out / WORLD), floor, scenario_files(item, tracks(read[item][1])))
        ScenarioView(out / WORLD / "scenarios" / item).load()
        return f"{WORLD}/scenarios/{item}"

    def import_item(self, item: str, out: Path) -> Path:
        target = super().import_item(item, out)
        write_attribution(target.parents[1] / ATTRIBUTION_FILE, self.attribution(item))
        return target

    def attribution(self, item: str) -> Attribution:
        return Attribution(source="MADRAS Fete des Lumieres 2022 TopView", url=f"https://zenodo.org/records/{RECORD}", license=LICENSE, authors=AUTHORS, citation=CITATION)
