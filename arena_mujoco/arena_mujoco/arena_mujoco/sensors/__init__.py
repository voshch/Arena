"""Sensor infrastructure for arena_mujoco."""

from __future__ import annotations

from . import camera, contact, imu, laser  # noqa: F401 (self-register via @register_sensor on import)
from .core import (
    SensorPublisher,
    get_renderer,
    install_sensor_hooks,
    register_sensor,
)

__all__ = [
    'SensorPublisher',
    'get_renderer',
    'install_sensor_hooks',
    'register_sensor',
]
