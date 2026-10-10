"""Pedestrian faces a camera can look through: REP-155 `face_<id>` frames and their view intrinsics."""

from __future__ import annotations

import math
import typing
from collections.abc import Iterable, Mapping

if typing.TYPE_CHECKING:
    from arena_people_msgs.msg import FaceView

TRACKED_SUFFIX = "/humans/faces/tracked"
FRAME_PREFIX = "face_"
VIEW_SUFFIX = "/view"


def tracked_topics(topics: Iterable[str]) -> list[str]:
    """The faces/tracked topics among graph topic names."""
    return sorted(name for name in topics if name.endswith(TRACKED_SUFFIX))


def view_topic(tracked: str, face_id: str) -> str:
    """The view topic of `face_id`, listed on the faces/tracked topic `tracked`."""
    return f"{tracked.removesuffix('/tracked')}/{face_id}{VIEW_SUFFIX}"


def frame(face_id: str) -> str:
    return f"{FRAME_PREFIX}{face_id}"


def face_id(name: str) -> str | None:
    """The face id of a `face_<id>` frame name, None for any other frame."""
    return name.removeprefix(FRAME_PREFIX) if name.startswith(FRAME_PREFIX) else None


def match(rosters: Mapping[str, Iterable[str]], face: str) -> tuple[str, str]:
    """(tracked topic, face id) for a face id, a `face_<id>` frame or an agent number unique across envs."""
    wanted = face_id(face) or face
    exact = [(topic, fid) for topic, ids in rosters.items() for fid in ids if fid == wanted]
    numbered = [(topic, fid) for topic, ids in rosters.items() for fid in ids if wanted.isdigit() and fid.endswith(f"_agent_{wanted}")]
    found = exact or numbered
    if len(found) == 1:
        return found[0]
    known = sorted(fid for ids in rosters.values() for fid in ids)
    if not found:
        raise LookupError(f"no face {face!r}, tracked faces: {known or 'none'}")
    raise LookupError(f"face {face!r} is ambiguous: {sorted(fid for _topic, fid in found)}")


def fov(view: FaceView) -> float:
    """Horizontal fov in rad of a face view's pinhole."""
    return 2.0 * math.atan(view.info.width / 2.0 / view.info.k[0])
