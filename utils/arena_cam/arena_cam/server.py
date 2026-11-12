"""`arena cam serve`: one env's camera server, playing the camera assets sent to `/arena/env_<n>/cam/play` on that env's surfaces and recording each labelled take."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import rclpy
import yaml
from arena_cam_msgs.srv import Play
from arena_rclpy_mixins.spin import run_main
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Trigger

from .camera import Camera
from .client import CamNode
from .record import PARTIAL, promote, record_root
from .surfaces import TargetSelection

SETUP_PARAM = "cam.setup"
SELECT_TIMEOUT = 60.0


class CamServer(CamNode):
    """Plays one take at a time, a new play cancels the running one. Rides an active lockstep run and never holds the sim."""

    ride_only = True

    def __init__(self, env_id: int, sim: bool, fps: float, setup: str, record_dir: Path | None = None) -> None:
        super().__init__(timeline=None, targets=TargetSelection(include_sim=sim, viz_all=False, viz_env=env_id), node_name=f"arena_cam_env_{env_id}", fps=fps, env=env_id)
        self.channel_env = f"/arena/env_{env_id}"
        self._selected = setup
        self._record_dir = record_dir
        self._unsynced_logged = False
        self._generator = f"{self.channel_env}/arena_env_node"
        self._pending: tuple[Camera, str] | None = None
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

    def _replace(self, pending: tuple[Camera, str] | None) -> None:
        self._pending = pending
        self._wake.set()
        if self._current is not None:
            self._current.cancel()

    async def _on_play(self, request: Play.Request, response: Play.Response) -> Play.Response:
        try:
            timeline = Camera(self._selection).load(request.root, yaml.safe_load(request.bundle) or {}, yaml.safe_load(request.params) or {})
        except (ValueError, KeyError, yaml.YAMLError) as e:
            response.message = str(e)
            return response
        self._replace((timeline, request.take))
        response.success = True
        return response

    async def _on_stop(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self._replace(None)
        response.success = True
        return response

    def _record_target(self, take: str) -> Path | None:
        """Where `take` records to, clearing what a cut take of that label left, None when it only plays: no record dir or label, no lockstep run, or a finished take of that label."""
        if self._record_dir is None or not take:
            return None
        if not self._scheduler_active:
            if not self._unsynced_logged:
                self.get_logger().error("cam: recording needs a lockstep run (lockstep:=true), playing only")
                self._unsynced_logged = True
            return None
        target = self._record_dir / f"{take}.mp4"
        if target.exists() or any(target.parent.glob(f"{target.stem}-*.mp4")):
            self.get_logger().error(f"cam: take {take} exists under {self._record_dir}, playing only")
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.stem}{PARTIAL}.mp4")
        for stale in partial.parent.glob(f"{partial.stem}*.mp4"):
            stale.unlink()
        return partial

    async def _take(self, timeline: Camera, take: str) -> None:
        try:
            if not await self.connect():
                return
            target = self._record_target(take)
            if target is None:
                await self.play(timeline)
                return
            self._record = target
            try:
                ok = await self.play(timeline)
            except asyncio.CancelledError:
                self.get_logger().warning(f"cam: take {take} cut short, partial files kept")
                raise
            finally:
                self._record = None
            if ok:
                promote(sorted(target.parent.glob(f"{target.stem}*.mp4")))
                self.get_logger().info(f"cam: saved take {take}")
            else:
                self.get_logger().warning(f"cam: take {take} incomplete, partial files kept")
        except Exception as e:
            self.fail(f"cam: take failed: {e}")

    async def _serve(self) -> None:
        """Run the latest take to its end or until a newer one cancels it, one at a time so a take's cleanup finishes first."""
        while rclpy.ok():
            await self._wake.wait()
            self._wake.clear()
            pending, self._pending = self._pending, None
            if pending is not None:
                self._current = asyncio.ensure_future(self._take(*pending))
                await asyncio.wait({self._current})
                self._current = None


def serve(env_id: int, sim: bool, fps: float, setup: str, record: str = "") -> None:
    """Run the camera server of env `env_id` until shutdown, recording under `record` when set."""
    record_dir = record_root(record, Path(os.environ.get("ARENA_DATA_DIR") or Path.cwd())) if record else None
    run_main(CamServer, env_id, sim, fps, setup, record_dir)
