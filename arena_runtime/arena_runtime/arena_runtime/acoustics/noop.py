"""Backend for acoustics:=none, nothing is launched and nothing is required."""

from arena_runtime.acoustics import BaseAcousticsSimulator


class NoopAcousticsSimulator(BaseAcousticsSimulator):
    @property
    def requires_map_server(self) -> bool:
        return False
