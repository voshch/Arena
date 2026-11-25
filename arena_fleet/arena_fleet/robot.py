import math
from collections.abc import Sequence

import attrs
from arena_robots.caps import MobileSpec


@attrs.frozen
class RobotFacts:
    """What the dispatcher takes from a robot model's mobile cap: body radius, top speed and laser reach."""

    radius_m: float
    max_speed_mps: float
    sensor_range_m: float
    sensor_fov_deg: float

    @classmethod
    def from_mobile(cls, mobile: MobileSpec) -> "RobotFacts":
        """From an instance's effective mobile cap."""
        laser = mobile.laser
        velocity = mobile.velocity_limits
        assert laser is not None and velocity is not None and mobile.radius is not None
        return cls(
            radius_m=mobile.radius,
            max_speed_mps=max(abs(velocity.linear.min), abs(velocity.linear.max)),
            sensor_range_m=laser.range,
            sensor_fov_deg=math.degrees(laser.angle.max - laser.angle.min),
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
