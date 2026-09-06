"""Backend for auditory:=none, nothing is launched and nothing is required."""

from task_generator.simulators.auditory import BaseAuditorySimulator


class NoopAuditorySimulator(BaseAuditorySimulator):
    @property
    def requires_map_server(self) -> bool:
        return False
