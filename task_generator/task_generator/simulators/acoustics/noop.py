"""Backend for acoustics:=none, nothing is launched and nothing is required."""

from task_generator.simulators.acoustics import BaseAcousticsSimulator


class NoopAcousticsSimulator(BaseAcousticsSimulator):
    @property
    def requires_map_server(self) -> bool:
        return False
