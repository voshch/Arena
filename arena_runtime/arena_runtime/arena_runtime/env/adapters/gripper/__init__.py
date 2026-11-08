from __future__ import annotations

from arena_runtime.env.adapters import ADAPTERS, Adapter


@ADAPTERS["gripper"].register("direct")
def _load_direct() -> type[Adapter]:
    from .direct import DirectGripperAdapter

    return DirectGripperAdapter


@ADAPTERS["gripper"].register("none")
def _load_none() -> type[Adapter]:
    from .none import NoneGripperAdapter

    return NoneGripperAdapter
