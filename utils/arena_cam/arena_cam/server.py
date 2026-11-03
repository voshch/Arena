"""`arena cam serve`: one env's camera server, playing the camera assets sent to `/arena/env_<n>/cam/play` on that env's surfaces."""

from __future__ import annotations

import asyncio

import rclpy
import yaml
from arena_cam_msgs.srv import Play
from arena_rclpy_mixins.spin import run_main
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Trigger

from .camera import Camera
from .client import CamNode
from .surfaces import TargetSelection

SETUP_PARAM = "cam.setup"
SELECT_TIMEOUT = 60.0


class CamServer(CamNode):
    """Plays one take at a time, a new play cancels the running one. Rides an active lockstep run and never holds the sim."""

    ride_only = True

    def __init__(self, env_id: int, sim: bool, fps: float, setup: str) -> None:
        super().__init__(timeline=None, targets=TargetSelection(include_sim=sim, viz_all=False, viz_env=env_id), node_name=f"arena_cam_env_{env_id}", fps=fps, env=env_id)
        self.channel_env = f"/arena/env_{env_id}"
        self._selected = setup
        self._generator = f"{self.channel_env}/arena_env_node"
        self._pending: Camera | None = None
        self._wake = asyncio.Event()
        self._current: asyncio.Future[None] | None = None
        self._worker: asyncio.Task[None] | None = None

    async def setup(self) -> None:
        self.create_service(Play, f"{self.channel_env}/cam/play", self._on_play)
        self.create_service(Trigger, f"{self.channel_env}/cam/stop", self._on_stop)
        self._worker = asyncio.ensure_future(self._serve())
        if self._selected:
            await self._select(self._selected)

    async def teardown(self) -> None:
        if self._selected:
            await self._set_setup("")

    async def _set_setup(self, request: str) -> bool:
        client = self.create_client_wrapper(SetParameters, f"{self._generator}/set_parameters", timeout=SELECT_TIMEOUT)
        if not await client.ensure(timeout_sec=SELECT_TIMEOUT):
            self.get_logger().warning(f"{self._generator} did not answer, cannot set {SETUP_PARAM}")
            return False
        value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=request)
        await client.call_timeout(SetParameters.Request(parameters=[Parameter(name=SETUP_PARAM, value=value)]))
        return True

    async def _select(self, request: str) -> None:
        """Have the env's task generator play `request` now and at every episode start."""
        if not await self._set_setup(request):
            return
        client = self.create_client_wrapper(Play, f"{self._generator}/cam/play", timeout=SELECT_TIMEOUT)
        if await client.ensure(timeout_sec=SELECT_TIMEOUT):
            response = await client.call_timeout(Play.Request(root=request))
            if response is not None and not response.success:
                self.get_logger().info(f"cam: {response.message}")

    def _replace(self, timeline: Camera | None) -> None:
        self._pending = timeline
        self._wake.set()
        if self._current is not None:
            self._current.cancel()

    async def _on_play(self, request: Play.Request, response: Play.Response) -> Play.Response:
        try:
            timeline = Camera(self._selection).load(request.root, yaml.safe_load(request.bundle) or {}, yaml.safe_load(request.params) or {})
        except (ValueError, KeyError, yaml.YAMLError) as e:
            response.message = str(e)
            return response
        self._replace(timeline)
        response.success = True
        return response

    async def _on_stop(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self._replace(None)
        response.success = True
        return response

    async def _take(self, timeline: Camera) -> None:
        try:
            if await self.connect():
                await self.play(timeline)
        except Exception as e:
            self.fail(f"cam: take failed: {e}")

    async def _serve(self) -> None:
        """Run the latest take to its end or until a newer one cancels it, one at a time so a take's cleanup finishes first."""
        while rclpy.ok():
            await self._wake.wait()
            self._wake.clear()
            timeline, self._pending = self._pending, None
            if timeline is not None:
                self._current = asyncio.ensure_future(self._take(timeline))
                await asyncio.wait({self._current})
                self._current = None


def serve(env_id: int, sim: bool, fps: float, setup: str) -> None:
    """Run the camera server of env `env_id` until shutdown."""
    run_main(CamServer, env_id, sim, fps, setup)
