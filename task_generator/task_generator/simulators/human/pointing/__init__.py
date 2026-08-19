"""Parametric PointAt animations for the Arena 36-DOF human wire contract."""

from .generator import (
    TEMPLATES,
    HoldPose,
    PointAtClip,
    PointAtGenerator,
    PointAtOptions,
    Template,
    angles_from_direction,
    direction_from_angles,
    load_template,
    point_at,
)
from .skeleton import Body, fk
from .table import BakedPointAt, PointTable, bake

__all__ = [
    "PointAtGenerator", "PointAtOptions", "PointAtClip", "HoldPose", "Template", "TEMPLATES",
    "point_at", "load_template", "direction_from_angles", "angles_from_direction",
    "BakedPointAt", "PointTable", "bake",
    "Body", "fk",
]
