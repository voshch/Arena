"""Per-sim launch defaults shared by the supervisor and task_generator.launch.py."""

DEFAULT_HUMAN: dict[str, str] = {
    "dummy": "dummy",
    "gazebo": "arena",
    "isaac": "arena",
    "mujoco": "arena",
}


def default_human(sim: str) -> str:
    """Human-backend default for a sim, unknown sims fall back to dummy."""
    return DEFAULT_HUMAN.get(sim, "dummy")
