import typing
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Orientation:
    x: float
    y: float
    z: float
    w: float

    @classmethod
    def parse(cls, obj: typing.Any) -> "Orientation":
        return cls(x=obj['x'], y=obj['y'], z=obj['z'], w=obj['w'])


@dataclass(frozen=True)
class Position:
    x: float
    y: float
    z: float

    @classmethod
    def parse(cls, obj: typing.Any) -> "Position":
        return cls(x=obj['x'], y=obj['y'], z=obj['z'])


@dataclass(frozen=True)
class Pose:
    position: Position
    orientation: Orientation

    @classmethod
    def parse(cls, obj: typing.Any) -> "Pose":
        """
        Parse PoseStamped message
        """
        return cls(
            position=Position.parse(obj['pose']['position']),
            orientation=Orientation.parse(obj['pose']['orientation']),
        )


@dataclass(frozen=True)
class Path:
    poses: typing.List[Pose]

    @classmethod
    def parse(cls, obj: typing.Any) -> "Path":
        return cls(poses=[Pose.parse(pose) for pose in obj["poses"]])


@dataclass(frozen=True)
class LaserScan:
    angle_min: float
    angle_max: float
    angle_increment: float
    time_increment: float
    scan_time: float
    range_min: float
    range_max: float
    ranges: np.ndarray
    intensities: typing.List[float]

    @classmethod
    def parse(cls, obj: typing.Any) -> "LaserScan":
        return cls(
            angle_min=obj['angle_min'],
            angle_max=obj['angle_max'],
            angle_increment=obj['angle_increment'],
            time_increment=obj['time_increment'],
            scan_time=obj['scan_time'],
            range_min=obj['range_min'],
            range_max=obj['range_max'],
            ranges=np.clip(np.array(obj['ranges'], dtype=float), obj['range_min'], obj['range_max']),
            intensities=obj['intensities'],
        )
