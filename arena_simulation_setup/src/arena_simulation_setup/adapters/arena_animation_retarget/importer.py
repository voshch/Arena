"""`arena asset import animation smplx`: not implemented. Convert SMPL-X motion with `ros2 run arena_simulation_setup animation_retarget` instead."""

from __future__ import annotations

from pathlib import Path

from arena_assets.importers import IMPORTERS, Attribution, Importer
from arena_assets.importers.providers import Http, Provider

NOT_IMPLEMENTED = "arena asset import animation smplx is not implemented, run `ros2 run arena_simulation_setup animation_retarget` instead"


class SMPLX(Importer):
    meta = IMPORTERS.meta("animation", "smplx")

    @classmethod
    def default_provider(cls) -> Provider:
        return Http("")

    def items(self) -> list[str]:
        print(NOT_IMPLEMENTED)
        return []

    def fetch(self, item: str) -> Path:
        raise RuntimeError(NOT_IMPLEMENTED)

    def convert(self, item: str, src: Path, out: Path) -> str:
        raise RuntimeError(NOT_IMPLEMENTED)

    def attribution(self, item: str) -> Attribution:
        raise RuntimeError(NOT_IMPLEMENTED)
