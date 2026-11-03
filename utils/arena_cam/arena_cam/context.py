"""Context tokens of a camera spec: `${env}`, `${robot}`, `${robot.<name>}`, `${human.<name>}`, `${face.<name>}`, filled per env."""

from __future__ import annotations

import asyncio
import re
import typing
from collections.abc import Callable, Iterable, Mapping

if typing.TYPE_CHECKING:
    import rclpy.node
    import rclpy.subscription
    from arena_people_msgs.msg import Pedestrians
    from arena_runtime_msgs.msg import RobotFleet

_TOKEN = re.compile(r"\$\{([^}]+)\}")

ROSTER_TIMEOUT = 10.0


def tokens(text: str) -> list[str]:
    """The context keys named in `text`."""
    return _TOKEN.findall(text)


def fill(text: str, value: Callable[[str], str]) -> str:
    """`text` with each token replaced by `value(key)`."""
    return _TOKEN.sub(lambda match: value(match.group(1)), text)


def human_id(humans: Mapping[str, int], name: str) -> int | None:
    """The id of the pedestrian `name`, whose roster name may carry env and level parts."""
    return next((ped_id for full, ped_id in humans.items() if name in full.split("/")), None)


def value(key: str, env_id: int, robots: Mapping[str, str], humans: Mapping[str, int]) -> str:
    """What a context token holds in one env, from its robots (name to base frame) and pedestrians (name to id), a LookupError naming what the env has."""
    if key == "env":
        return f"env_{env_id}"
    kind, _, name = key.partition(".")
    if kind == "robot":
        if not robots:
            raise LookupError(f"${{{key}}}: env {env_id} has no robots")
        if not name:
            return next(iter(robots.values()))
        if name in robots:
            return robots[name]
        raise LookupError(f"${{{key}}}: env {env_id} has robots {sorted(robots)}")
    if kind in ("human", "face"):
        ped_id = human_id(humans, name)
        if ped_id is None:
            raise LookupError(f"${{{key}}}: env {env_id} has humans {sorted(humans)}")
        return f"body_env_{env_id}_agent_{ped_id}" if kind == "human" else f"env_{env_id}_agent_{ped_id}"
    raise LookupError(f"${{{key}}}: not a context token")


class EnvContext:
    """The live fleet and pedestrian roster of one env, subscribed on the first token that needs them."""

    def __init__(self, node: rclpy.node.Node, env_id: int) -> None:
        self._node = node
        self._env_id = env_id
        self._robots: dict[str, str] | None = None
        self._humans: dict[str, int] | None = None
        self._subscriptions: list[rclpy.subscription.Subscription] = []

    def _watch(self) -> None:
        from arena_people_msgs.msg import Pedestrians
        from arena_rclpy_mixins.qos import latched
        from arena_runtime_msgs.msg import RobotFleet

        ns = f"/arena/env_{self._env_id}"
        self._subscriptions = [
            self._node.create_subscription(RobotFleet, f"{ns}/arena_env_node/state/robots", self._on_fleet, latched(1)),
            self._node.create_subscription(Pedestrians, f"{ns}/arena_peds", self._on_peds, 10),
        ]

    def _on_fleet(self, msg: RobotFleet) -> None:
        self._robots = {robot.descriptor.name: robot.descriptor.base_frame for robot in msg.robots}

    def _on_peds(self, msg: Pedestrians) -> None:
        self._humans = {ped.name: ped.id for ped in msg.pedestrians}

    def _ready(self, keys: Iterable[str]) -> bool:
        kinds = {key.partition(".")[0] for key in keys}
        return (self._robots is not None or "robot" not in kinds) and (self._humans is not None or not kinds & {"human", "face"})

    async def fill(self, text: str) -> str:
        """`text` with its tokens filled, waiting for the roster they read, a LookupError when a token has no value."""
        keys = tokens(text)
        if not self._subscriptions and keys != ["env"]:
            self._watch()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + ROSTER_TIMEOUT
        while not self._ready(keys) and loop.time() < deadline:
            await asyncio.sleep(0.05)
        return fill(text, lambda key: value(key, self._env_id, self._robots or {}, self._humans or {}))
