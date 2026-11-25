"""Walks of the TF tree a tf2 buffer holds."""

from __future__ import annotations

import math

import rclpy.time
import tf2_ros
import yaml


def frames_below(buffer: tf2_ros.Buffer, frame: str) -> list[str]:
    """The TF frame and every frame below it."""
    children: dict[str, list[str]] = {}
    for name, info in (yaml.safe_load(buffer.all_frames_as_yaml()) or {}).items():
        children.setdefault(str(info.get('parent', '')), []).append(str(name))
    found = [frame]
    for parent in found:
        found.extend(children.get(parent, ()))
    return found


def reach_below(buffer: tf2_ros.Buffer, frame: str) -> float:
    """Distance from the TF frame to the farthest frame below it, 0 without any."""
    farthest = 0.0
    for below in frames_below(buffer, frame)[1:]:
        try:
            t = buffer.lookup_transform(frame, below, rclpy.time.Time()).transform.translation
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            continue
        farthest = max(farthest, math.hypot(t.x, t.y, t.z))
    return farthest
