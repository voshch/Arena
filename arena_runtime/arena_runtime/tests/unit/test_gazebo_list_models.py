"""`gz model --list` against an unreachable sim raises SimUnavailable."""

from __future__ import annotations

import asyncio
import os
import shutil

import pytest

pytest.importorskip("arena_runtime.sim.gazebo_simulator.gazebo_simulator")

from arena_runtime.sim import SimUnavailable  # noqa: E402
from arena_runtime.sim.gazebo_simulator.gazebo_simulator import GazeboHost  # noqa: E402


@pytest.mark.skipif(shutil.which("gz") is None, reason="gz CLI not installed")
def test_list_models_raises_when_sim_unreachable() -> None:
    previous = os.environ.get("GZ_PARTITION")
    os.environ["GZ_PARTITION"] = f"no_sim_{os.getpid()}"
    try:
        with pytest.raises(SimUnavailable):
            asyncio.run(GazeboHost._list_models(None))
    finally:
        if previous is None:
            del os.environ["GZ_PARTITION"]
        else:
            os.environ["GZ_PARTITION"] = previous
