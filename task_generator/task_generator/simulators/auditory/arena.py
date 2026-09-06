"""Backend for auditory:=arena, the arena_auditory sidecar (propagation, hearing, emission, playback)."""

from task_generator.simulators.auditory import BaseAuditorySimulator


class ArenaAuditorySimulator(BaseAuditorySimulator):
    """The propagation node builds its acoustic scene from the map topic, so the map server must be up before episodes start."""

    @property
    def requires_map_server(self) -> bool:
        return True
