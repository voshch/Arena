from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

Action = typing.Literal["load", "configure", "activate", "wait", "fail"]
Transition = tuple[Action, tuple[str, ...]]

_KNOWN = frozenset({"unconfigured", "inactive", "active", "finalized"})


def next_transition(expected: Sequence[str], states: Mapping[str, str]) -> Transition | None:
    """The one controller_manager call the snapshot needs next, None once all are active."""
    absent: list[str] = []
    unconfigured: list[str] = []
    inactive: list[str] = []
    transient: list[str] = []
    finalized: list[str] = []
    for name in expected:
        label = states.get(name)
        if label is None:
            absent.append(name)
        elif label == "active":
            continue
        elif label == "unconfigured":
            unconfigured.append(name)
        elif label == "inactive":
            inactive.append(name)
        elif label == "finalized":
            finalized.append(name)
        elif label not in _KNOWN:
            transient.append(name)
    if finalized:
        return ("fail", tuple(finalized))
    if absent:
        return ("load", (absent[0],))
    if unconfigured:
        return ("configure", (unconfigured[0],))
    if transient:
        return ("wait", tuple(transient))
    if inactive:
        return ("activate", tuple(inactive))
    return None
