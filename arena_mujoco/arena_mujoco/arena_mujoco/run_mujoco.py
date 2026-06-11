"""MuJoCo simulation server entry point."""

import faulthandler
import os
import signal
import sys
import threading
import time

os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
import rclpy.node  # noqa: E402
import rosgraph_msgs.msg  # noqa: E402
import std_srvs.srv  # noqa: E402
from arena_mujoco_msgs.srv import DeletePrims, ResetWorld, Step  # noqa: E402
from arena_people_msgs.msg import Pedestrians  # noqa: E402
from rclpy.executors import ExternalShutdownException  # noqa: E402

import arena_mujoco.control  # noqa: F401, E402 (registers pre_compile + post_spawn + pump hooks at import time)
import arena_mujoco.odom  # noqa: F401, E402 (registers post_spawn + pump hooks at import time)
import arena_mujoco.sensors  # noqa: E402
from arena_mujoco import context, hooks  # noqa: E402
from arena_mujoco.scene import PHYSICS_DT, SceneStore  # noqa: E402
from arena_mujoco.services import services  # noqa: E402
from arena_mujoco.services.SpawnCeilings import CEILING_GROUP  # noqa: E402
from arena_mujoco.services.SpawnPedestrians import release_pedestrians, reset_env, update_pedestrians  # noqa: E402
from arena_mujoco.services.SpawnUrdf import forget_robots  # noqa: E402
from arena_mujoco.services.utils import SERVICE_QOS  # noqa: E402

try:
    import mujoco.viewer as _mj_viewer
except Exception:  # noqa: BLE001 - a missing GL stack must not break headless runs
    _mj_viewer = None

_STEP_PERIOD_S = 0.02

_VIEWER_PERIOD_S = 1.0 / 30.0

_STEPS_PER_TICK = int(_STEP_PERIOD_S / PHYSICS_DT)

_VIEWER_ENV = 0
_VIEWER_ELEVATION_DEG = -50.0
_VIEWER_MIN_DISTANCE_M = 4.0
_PARKED_BELOW_Z = -100.0


def _arg_bool(name: str, default: bool) -> bool:
    """Parse a boolean CLI argument in --name value or name:=value form."""
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1].lower() in ('true', '1')
    prefix = f"{name.lstrip('-')}:="
    for arg in sys.argv:
        if arg.startswith(prefix):
            return arg[len(prefix) :].lower() in ('true', '1')
    return default


class MujocoController(rclpy.node.Node):
    """ROS 2 node that exposes MuJoCo simulation control services."""

    def __init__(self, scene_store: SceneStore, headless: bool = False) -> None:
        super().__init__(node_name='mujoco')

        self._scene_store = scene_store
        self._paused = False
        self._headless = headless or _mj_viewer is None
        self._viewer = None
        self._viewer_loaded: tuple[mujoco.MjModel, mujoco.MjData] | None = None
        self._viewer_frame: tuple[mujoco.MjModel, mujoco.MjData, int] | None = None
        self._viewer_edits = 0
        self._viewer_view: tuple[tuple[float, ...], float] | None = None
        self._viewer_idle = threading.Event()
        self._viewer_due = 0.0
        self._viewer_wake = threading.Event()
        self._viewer_thread: threading.Thread | None = None

        self.__pause_srv = self.create_service(
            std_srvs.srv.Trigger,
            'mujoco/PauseSimulation',
            self._cb_pause,
            qos_profile=SERVICE_QOS,
        )
        self.__unpause_srv = self.create_service(
            std_srvs.srv.Trigger,
            'mujoco/UnpauseSimulation',
            self._cb_unpause,
            qos_profile=SERVICE_QOS,
        )
        self.__step_srv = self.create_service(
            Step,
            'mujoco/Step',
            self._cb_step,
            qos_profile=SERVICE_QOS,
        )
        self.__reset_world_srv = self.create_service(
            ResetWorld,
            'mujoco/ResetWorld',
            self._cb_reset_world,
            qos_profile=SERVICE_QOS,
        )
        self.__delete_prims_srv = self.create_service(
            DeletePrims,
            'mujoco/DeletePrims',
            self._cb_delete_prims,
            qos_profile=SERVICE_QOS,
        )

        for service in services:
            service.create(self)

        self._peds_sub = self.create_subscription(Pedestrians, '/mujoco/arena_peds', self._cb_peds, 10)
        self._clock_pub = self.create_publisher(rosgraph_msgs.msg.Clock, '/clock', 10)
        self.create_timer(_STEP_PERIOD_S, self._tick)

    def _cb_pause(
        self,
        request: std_srvs.srv.Trigger.Request,
        response: std_srvs.srv.Trigger.Response,
    ) -> std_srvs.srv.Trigger.Response:
        self._paused = True
        response.success = True
        return response

    def _cb_unpause(
        self,
        request: std_srvs.srv.Trigger.Request,
        response: std_srvs.srv.Trigger.Response,
    ) -> std_srvs.srv.Trigger.Response:
        self._paused = False
        response.success = True
        return response

    def _cb_step(
        self,
        request: Step.Request,
        response: Step.Response,
    ) -> Step.Response:
        self._scene_store.compile_dirty()
        self._advance(int(request.n))
        response.success = True
        response.sim_time = self._scene_store.time
        return response

    def _cb_peds(self, msg: Pedestrians) -> None:
        update_pedestrians(msg.pedestrians)

    def _cb_reset_world(
        self,
        request: ResetWorld.Request,
        response: ResetWorld.Response,
    ) -> ResetWorld.Response:
        env_id = int(request.env_id)
        self._scene_store.ensure_env(env_id)
        forget_robots(env_id, '')
        self._scene_store.remove_by_prefix(env_id, '')
        reset_env(env_id)
        self._scene_store.recompile(env_id)
        response.ret = True
        return response

    def _cb_delete_prims(
        self,
        request: DeletePrims.Request,
        response: DeletePrims.Response,
    ) -> DeletePrims.Response:
        env_id = int(request.env_id)
        self._scene_store.ensure_env(env_id)
        ret: list[bool] = []
        for name in request.names:
            forget_robots(env_id, name)
            if name.endswith('/'):
                release_pedestrians(name)
                ret.append(self._scene_store.remove_by_prefix(env_id, name[:-1]) > 0)
            else:
                ret.append(self._scene_store.remove(env_id, name))
        response.ret = ret
        return response

    def _tick(self) -> None:
        self._scene_store.compile_dirty()
        if self._paused:
            hooks.run_pumps()
        else:
            self._advance(_STEPS_PER_TICK)
        self._update_viewer()

    def _advance(self, n: int) -> None:
        """Step every env n physics steps, draining the pumps before publishing /clock for each tick."""
        while n > 0:
            chunk = min(n, _STEPS_PER_TICK)
            self._scene_store.step(chunk)
            hooks.run_pumps()
            self._publish_clock()
            n -= chunk

    def _publish_clock(self) -> None:
        sim_time = self._scene_store.time
        msg = rosgraph_msgs.msg.Clock()
        msg.clock.sec = int(sim_time)
        msg.clock.nanosec = int(round((sim_time - int(sim_time)) * 1e9))
        self._clock_pub.publish(msg)

    def _update_viewer(self) -> None:
        """Hand env 0's state to the native passive viewer at most every _VIEWER_PERIOD_S wall seconds, skipping while it still draws the last one."""
        if self._headless:
            return
        now = time.monotonic()
        if now < self._viewer_due:
            return
        self._viewer_due = now + _VIEWER_PERIOD_S
        model = self._scene_store.get_model(_VIEWER_ENV)
        data = self._scene_store.get_data(_VIEWER_ENV)
        if model is None or data is None:
            return
        if self._viewer is None:
            frame = (model, mujoco.MjData(model))
            try:
                self._viewer = _mj_viewer.launch_passive(*frame)
            except Exception as exc:  # noqa: BLE001 - no display, fall back to headless
                self.get_logger().warn(f'MuJoCo viewer unavailable, continuing headless: {exc}')
                self._headless = True
                return
            self._viewer.opt.geomgroup[CEILING_GROUP] = 0
            self._viewer_loaded = frame
            self._viewer_frame = (*frame, self._scene_store.model_edits(_VIEWER_ENV))
            self._viewer_edits = self._viewer_frame[2]
            self._viewer_idle.set()
            self._viewer_thread = threading.Thread(target=self._draw_frames, name='mj_viewer', daemon=True)
            self._viewer_thread.start()
            self.get_logger().info('MuJoCo passive viewer attached to env 0')
        if not self._viewer.is_running():
            self._close_viewer()
            self._headless = True
            return
        if not self._viewer_idle.is_set():
            return
        shown = self._viewer_frame[1] if self._viewer_frame[0] is model else mujoco.MjData(model)
        mujoco.mj_copyData(shown, model, data)
        self._viewer_frame = (model, shown, self._scene_store.model_edits(_VIEWER_ENV))
        self._viewer_idle.clear()
        self._viewer_wake.set()

    def _draw_frames(self) -> None:
        """Viewer thread: show each handed-over frame, loading the model first when a recompile swapped it."""
        while True:
            self._viewer_wake.wait()
            self._viewer_wake.clear()
            viewer = self._viewer
            if viewer is None:
                return
            model, shown, edits = self._viewer_frame
            if self._viewer_loaded[0] is not model:
                sim = viewer._get_sim()  # noqa: SLF001 - the only entry to swap a passive viewer's model
                if sim is None:
                    return
                sim.load(model, shown, '')
                self._viewer_loaded = (model, shown)
                self._viewer_edits = edits
                self._frame_world(viewer, model, shown)
            viewer.sync(state_only=edits == self._viewer_edits)
            self._viewer_edits = edits
            self._viewer_idle.set()

    def _frame_world(self, viewer: object, model: mujoco.MjModel, shown: mujoco.MjData) -> None:
        """Point the free camera at everything standing in the env, unless the user moved it since the last framing."""
        camera = viewer.cam
        if self._viewer_view is not None and (tuple(camera.lookat), camera.distance) != self._viewer_view:
            return
        standing = (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE) & (shown.geom_xpos[:, 2] > _PARKED_BELOW_Z)
        if not standing.any():
            return
        low = shown.geom_xpos[standing].min(axis=0)
        high = shown.geom_xpos[standing].max(axis=0)
        with viewer.lock():
            camera.lookat[:] = (low + high) / 2.0
            camera.distance = max(float(np.linalg.norm((high - low)[:2])), _VIEWER_MIN_DISTANCE_M)
            camera.elevation = _VIEWER_ELEVATION_DEG
        self._viewer_view = (tuple(camera.lookat), camera.distance)

    def _close_viewer(self) -> None:
        viewer, self._viewer = self._viewer, None
        self._viewer_wake.set()
        if self._viewer_thread is not None:
            self._viewer_thread.join(timeout=2.0)
            self._viewer_thread = None
        if viewer is not None:
            viewer.close()
        self._viewer_loaded = None
        self._viewer_frame = None
        self._viewer_view = None

    def destroy_node(self) -> None:
        self._close_viewer()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    """Initialize MuJoCo server node and spin."""
    headless = _arg_bool('--headless', False)

    faulthandler.register(signal.SIGUSR1, all_threads=True)

    scene_store = SceneStore()
    context.set_scene_store(scene_store)

    rclpy.init(args=[])
    controller = MujocoController(scene_store, headless=headless)
    hooks.set_node(controller)
    arena_mujoco.sensors.install_sensor_hooks()

    try:
        rclpy.spin(controller)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        controller.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
