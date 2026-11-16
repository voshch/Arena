import math
import typing
from collections.abc import Mapping, Sequence

import attrs


@attrs.frozen
class RobotFacts:
    """What the dispatcher takes from a robot model's mobile cap: body radius, top speed and laser reach."""

    radius_m: float
    max_speed_mps: float
    sensor_range_m: float
    sensor_fov_deg: float

    @classmethod
    def from_mobile(cls, mobile: Mapping[str, typing.Any]) -> "RobotFacts":
        """From the mapping of caps/mobile.yaml."""
        laser = mobile["laser"]
        return cls(
            radius_m=float(mobile["radius"]),
            max_speed_mps=max(abs(float(v)) for v in mobile["velocity_limits"]["linear"]),
            sensor_range_m=float(laser["range"]),
            sensor_fov_deg=math.degrees(float(laser["angle"]["max"]) - float(laser["angle"]["min"])),
        )

    @classmethod
    def fleet(cls, robots: Sequence["RobotFacts"]) -> "RobotFacts":
        """The fleet as one robot: the largest body, the slowest top speed and the shortest, narrowest laser."""
        return cls(
            radius_m=max(r.radius_m for r in robots),
            max_speed_mps=min(r.max_speed_mps for r in robots),
            sensor_range_m=min(r.sensor_range_m for r in robots),
            sensor_fov_deg=min(r.sensor_fov_deg for r in robots),
        )
