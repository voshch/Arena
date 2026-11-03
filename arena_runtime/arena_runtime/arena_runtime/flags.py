"""Namespaced launch flags (`debug.*`, `optim.*`): expansion and lookup.

Two equivalent input syntaxes collapse onto the same canonical dotted params:
`<ns>:=a,b` is shorthand for `<ns>.a:=1 <ns>.b:=1`. The dotted form is canonical
and wins on conflict; the bare comma list is pure sugar expanded at launch time.
Flags are open-namespaced (any token is accepted) and read back by prefix.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from typing import TYPE_CHECKING

from rclpy.parameter import Parameter

if TYPE_CHECKING:
    import launch
    import rclpy.node

_FALSY = frozenset({"", "0", "false", "no", "off"})


def truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in _FALSY
    return bool(value)


def flag_enabled(node: rclpy.node.Node, namespace: str, flag: str) -> bool:
    """True if `<namespace>.<flag>` is declared and set to a truthy value."""
    param = node.get_parameters_by_prefix(namespace).get(flag)
    return isinstance(param, Parameter) and truthy(param.value)


class ObstaclesOptim(enum.IntEnum):
    """`optim.obstacles` level, accepts the number or the alias (`bbox`, `2`, ...)."""

    FULL = 0  # full mesh asset
    BBOX = 1  # bounding-box primitive where annotated, else full mesh
    NONE = 2  # skip obstacle spawning entirely

    @classmethod
    def coerce(cls, value: object) -> ObstaclesOptim:
        if isinstance(value, bool):
            return cls.BBOX if value else cls.FULL
        if isinstance(value, (int, float)):
            return cls(min(max(int(value), cls.FULL), cls.NONE))
        if isinstance(value, str):
            key = value.strip().lower()
            alias = _OBSTACLES_ALIASES.get(key)
            if alias is not None:
                return alias
            if key.lstrip("-").isdigit():
                return cls.coerce(int(key))
        return cls.FULL


_OBSTACLES_ALIASES = {
    "": ObstaclesOptim.FULL,
    "full": ObstaclesOptim.FULL,
    "bbox": ObstaclesOptim.BBOX,
    "none": ObstaclesOptim.NONE,
    "no_obstacles": ObstaclesOptim.NONE,
}


def obstacles_optim_level(node: rclpy.node.Node) -> ObstaclesOptim:
    """Resolve `optim.obstacles`, honouring the deprecated `optim.no_obstacles` flag as NONE."""
    params = node.get_parameters_by_prefix("optim")
    obstacles = params.get("obstacles")
    if isinstance(obstacles, Parameter):
        return ObstaclesOptim.coerce(obstacles.value)
    deprecated = params.get("no_obstacles")
    if isinstance(deprecated, Parameter) and truthy(deprecated.value):
        return ObstaclesOptim.NONE
    return ObstaclesOptim.FULL


class MapSource(enum.Enum):
    """`debug.map_source`: occupancy map origin."""

    COMPUTE = "compute"  # rasterize from world.yaml (default)
    DISK = "disk"  # on-disk map.png is the single source of truth

    @classmethod
    def coerce(cls, value: object) -> MapSource:
        if isinstance(value, str):
            key = value.strip().lower()
            for member in cls:
                if member.value == key:
                    return member
        return cls.COMPUTE


def map_source(node: rclpy.node.Node) -> MapSource:
    """Resolve `debug.map_source` (default COMPUTE)."""
    param = node.get_parameters_by_prefix("debug").get("map_source")
    if isinstance(param, Parameter):
        return MapSource.coerce(param.value)
    return MapSource.COMPUTE


def expand_flag_namespace(
    context: launch.LaunchContext,
    name: str,
    coerce: Callable[[str], object],
) -> dict[str, object]:
    """Expand `<name>:=a,b` shorthand into dotted params `{name.a: 1, name.b: 1}`,
    overlaid by any explicit `<name>.x:=v` (coerced; explicit wins)."""
    configs = context.launch_configurations
    out: dict[str, object] = {f"{name}.{token}": 1 for raw in configs.get(name, "").split(",") if (token := raw.strip())}
    prefix = f"{name}."
    for key, value in configs.items():
        if key.startswith(prefix) and key[len(prefix) :].strip():
            out[key] = coerce(value)
    return out
