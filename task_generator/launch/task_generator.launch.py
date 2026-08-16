import atexit
import contextlib
import os
import tempfile
import time

import launch
import launch.event_handlers
import launch.substitutions
import launch_ros.actions
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_bringup.actions import IsolatedGroupAction
from arena_rclpy_mixins import launch_str_to_value
from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction
from arena_bringup.defaults import default_human
from arena_bringup.substitutions import LaunchArgument
from task_generator.utils.flags import expand_flag_namespace, truthy
from launch.actions import (
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

_REGISTER_RETRY_SEC = 1.0
_REGISTER_LOG_INTERVAL_SEC = 10.0
_AUTO_ENV_ID = 0xFFFF


def _allocate_env(env_id: int, ns: str) -> tuple[int, str, str]:
    import rclpy
    from arena_runtime_msgs.srv import RegisterEnv
    from rclpy.node import Node

    if not rclpy.ok():
        rclpy.init(args=[])

    node = Node(f"task_generator_launch_{os.getpid()}_{env_id}")
    logger = node.get_logger()
    try:
        cli = node.create_client(RegisterEnv, "/arena/register_env")

        start = time.monotonic()
        next_log = start + _REGISTER_LOG_INTERVAL_SEC
        while not cli.wait_for_service(timeout_sec=_REGISTER_RETRY_SEC):
            now = time.monotonic()
            if now >= next_log:
                logger.warning(f"waiting for /arena/register_env ({int(now - start)}s elapsed)")
                next_log = now + _REGISTER_LOG_INTERVAL_SEC

        req = RegisterEnv.Request()
        req.caller_id = f"/task_generator_launch_{os.getpid()}"
        req.env_id = env_id
        req.ns = ns

        while True:
            future = cli.call_async(req)
            start = time.monotonic()
            next_log = start + _REGISTER_LOG_INTERVAL_SEC
            while not future.done():
                rclpy.spin_until_future_complete(node, future, timeout_sec=_REGISTER_RETRY_SEC)
                now = time.monotonic()
                if not future.done() and now >= next_log:
                    logger.warning(f"waiting for /arena/register_env response ({int(now - start)}s elapsed)")
                    next_log = now + _REGISTER_LOG_INTERVAL_SEC
            resp = future.result()
            if resp.success:
                return (resp.env_id, resp.ns, resp.sim)
            if "not ACTIVE" in resp.error_msg:
                time.sleep(_REGISTER_RETRY_SEC)
                continue
            raise RuntimeError(f"/arena/register_env failed: {resp.error_msg}")
    finally:
        node.destroy_node()


def generate_launch_description():
    bringup_dir = get_package_share_directory("arena_bringup")

    ld_items = []
    LaunchArgument.auto_append(ld_items)

    log_level = LaunchArgument(
        name='log_level',
        default_value='warn',
        description='Per-node log level. See launch README.',
    )

    env_id = LaunchArgument(
        name="env_id",
        default_value=str(_AUTO_ENV_ID),
        description="Requested env id; 65535 = auto-allocate via /arena/register_env.",
    )

    managed = LaunchArgument(
        name="managed",
        default_value="false",
        description="true = arena pre-reserved; skip /arena/register_env and use env_id/ns from args. Placement comes via confirm_world either way.",
    )

    ns = LaunchArgument(
        name="ns",
        default_value="",
        description="Explicit ns path (e.g. for sim2real); empty = auto-generate.",
    )

    sim = LaunchArgument(
        name="sim",
        default_value="",
        description="empty = adopt the runtime's sim; explicit [dummy, gazebo, isaac] must match the runtime",
    )
    # human/mobile defaults derive from arena's authoritative `sim` (the RegisterEnv
    # response, or the sim arg arena passes for managed envs). Empty here means
    # "use arena_sim". User can still override by passing e.g. human:=hunav explicitly.
    human = LaunchArgument(
        name="human",
        default_value="",
        description="empty = derive from arena_sim ({dummy: manual, gazebo|isaac: arena})",
    )
    auditory = LaunchArgument(
        name="auditory",
        choices=["none", "arena"],
        default_value="none",
        description="Auditory pipeline: none, or arena (propagation, robot hearing, robot sound, human sound).",
    )
    auditory_viz = LaunchArgument(
        name="auditory.viz",
        default_value="false",
        description="Publish source/portal/listener propagation markers.",
    )
    auditory_playback = LaunchArgument(
        name="auditory.playback",
        default_value="auto",
        description="PortAudio output device for workstation playback; auto = system default, none = no playback nodes.",
    )
    auditory_block_size = LaunchArgument(
        name="auditory.block_size",
        default_value="2048",
        description="Audio callback block size; raise on repeated PulseAudio underflows.",
    )
    auditory_assets = LaunchArgument(
        name="auditory.assets",
        default_value=PathJoinSubstitution([
            FindPackageShare("task_generator"),
            "config", "auditory", "acoustic_assets.yaml",
        ]),
        description="Acoustic asset catalog used by all playback nodes.",
    )
    auditory_sound_dir = LaunchArgument(
        name="auditory.sound_dir",
        default_value=PathJoinSubstitution([
            FindPackageShare("task_generator"),
            "sounds",
        ]),
        description="Directory containing WAV files named by the catalog.",
    )
    auditory_propagation = LaunchArgument(
        name="auditory.propagation",
        choices=["level3", "pyroomacoustics"],
        default_value="pyroomacoustics",
        description="Propagation backend.",
    )
    auditory_multi_portal = LaunchArgument(
        name="auditory.multi_portal",
        default_value="true",
        description="Allow pyroomacoustics RIR rendering across multi-hop door/opening portal routes.",
    )
    auditory_rir_in_propagation = LaunchArgument(
        name="auditory.rir_in_propagation",
        default_value="true",
        description="Compute RIR metadata in the propagation node instead of deferring to playback.",
    )
    auditory_robot_sound = LaunchArgument(
        name="auditory.robot_sound",
        default_value="true",
        description="Let robots emit motor audio (robots stay listeners regardless).",
    )
    auditory_motor = LaunchArgument(
        name="auditory.motor",
        choices=["off", "wav", "procedural"],
        default_value="procedural",
        description="Motor audio: off, WAV sequence, or calibrated procedural synthesis (Jackal; other models use WAVs).",
    )
    auditory_motor_playback = LaunchArgument(
        name="auditory.motor.playback",
        choices=["sequence", "single_loop"],
        default_value="sequence",
        description="WAV motor audio: start/loop/stop sequence, or a single repeating loop.",
    )
    auditory_environment_playback = LaunchArgument(
        name="auditory.environment_playback",
        default_value="true",
        description="Play propagated environment audio locally; emission and robot hearing continue when false.",
    )
    auditory_static_devices = LaunchArgument(
        name="auditory.static_devices",
        default_value="[]",
        description="YAML list of world-independent environment audio systems (radios, alarms); non-empty enables the audio_systems module.",
    )
    auditory_listener = LaunchArgument(
        name="auditory.listener",
        default_value="",
        description="Microphone ID that feeds playback (e.g. robot1_mic); RViz selects it when empty.",
    )
    auditory_microphones = LaunchArgument(
        name="auditory.microphones",
        default_value="[]",
        description="YAML list of robot microphone mappings (owner, robot, placement, frame, index).",
    )
    auditory_viewport_height = LaunchArgument(
        name="auditory.viewport_height",
        default_value="1.6",
        description="Listening height of the viewport camera's down-projection microphone.",
    )
    robot = LaunchArgument(name="robot", default_value="auto")
    tm_robots = LaunchArgument(name="tm_robots", default_value="explore")
    task_config = LaunchArgument(name="task_config", default_value="")
    episodes = LaunchArgument(
        name='episodes',
        default_value='-1',
        description='Stop the env after N episodes (-1 = run forever).',
    )
    scenario_file = LaunchArgument(
        name='scenario_file',
        default_value='',
        description='Sets task.scenario.file ROS param (empty = use parameter_file default).',
    )
    tm_obstacles = LaunchArgument(name="tm_obstacles", default_value="random")
    tm_modules = LaunchArgument(name="tm_modules", default_value="rviz_ui")
    LaunchArgument(name="optim", default_value=os.environ.get("ARENA_OPTIM", ""))
    world = LaunchArgument(name="world", default_value="map_empty")
    mobile = LaunchArgument(
        name="mobile",
        default_value="",
        description="mobile adapter kind; empty = derive from arena_sim ({dummy: none, *: nav2})",
    )
    planner = LaunchArgument(
        name="planner",
        default_value="",
        description="top-level planner selector; resolves to mobile:=<adapter> mobile.<selector>:=<name> via arena_planners.resolver",
    )
    arm = LaunchArgument(
        name="arm",
        default_value="moveit",
        description="arm adapter kind",
    )
    record_data_dir = LaunchArgument(name="record_data_dir", default_value="")
    disable_auto_recorder = LaunchArgument(name="disable_auto_recorder", default_value="false")
    LaunchArgument(
        name="debug",
        default_value="",
        description="comma list of debug tokens (e.g. aiomonitor,map_server); also debug.<token>:=true",
    )
    debug = LaunchArgument(name="debug", default_value="False")
    auto_reset = LaunchArgument(
        name="auto_reset",
        default_value="true",
        description=("true = standalone: node auto-advances episodes. false = managed: external controller drives resets via lifecycle/reset_episode."),
    )
    fail_on_collision = LaunchArgument(
        name="fail_on_collision",
        default_value="false",
        description="true = abort the episode (FAILED) when the robot footprint contacts a wall, static obstacle, or pedestrian.",
    )
    train_mode = LaunchArgument(name="train_mode", default_value="false")
    parameter_file = LaunchArgument(
        name="parameter_file",
        default_value=os.path.join(bringup_dir, "configs", "task_generator.yaml"),
    )

    def _build_env_actions(
        allocated_id: int,
        allocated_ns: str,
        arena_sim: str,
        context: launch.LaunchContext,
    ) -> list[launch.LaunchDescriptionEntity]:
        prefix_val = f"env_{allocated_id}"

        _label = f"arena env_{allocated_id}"
        with contextlib.suppress(OSError), open("/dev/tty", "w") as tty:
            tty.write(f"\033]0;{_label}\007\033]30;{_label}\007")
            tty.flush()

        def _restore_terminal_titles():
            with contextlib.suppress(OSError), open("/dev/tty", "w") as tty:
                tty.write("\033]0;\007\033]30;\007")
                tty.flush()

        atexit.register(_restore_terminal_titles)

        human_val = launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(human.substitution)) or default_human(arena_sim)
        mobile_val = launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(mobile.substitution)) or {"dummy": "none"}.get(arena_sim, "nav2")
        arm_val = launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(arm.substitution))
        tm_modules_val = launch.utilities.perform_substitutions(
            context,
            launch.utilities.normalize_to_list_of_substitutions(
                tm_modules.substitution
            ),
        )
        configured_modules = [
            value.strip()
            for value in tm_modules_val.split(",")
            if value.strip()
        ]
        static_devices_val = launch.utilities.perform_substitutions(
            context,
            launch.utilities.normalize_to_list_of_substitutions(
                auditory_static_devices.substitution
            ),
        ).strip()
        static_audio_enabled = static_devices_val not in ("", "[]")
        if static_audio_enabled and "audio_systems" not in configured_modules:
            configured_modules.append("audio_systems")
        tm_modules_val = ",".join(configured_modules)

        planner_val = launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(planner.substitution))
        _planner_selector_override: tuple[str, str] | None = None
        if planner_val:
            try:
                from arena_planners.resolver import ResolverError, resolve
            except ImportError as exc:
                raise RuntimeError(f"arena_planners is not importable ({exc}); run `arena build --packages-select arena_planners` first") from exc
            try:
                resolved = resolve(planner_val)
            except ResolverError as exc:
                raise RuntimeError(str(exc)) from exc
            explicit_mobile = launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(mobile.substitution))
            if explicit_mobile and explicit_mobile != resolved.adapter_kind:
                launch.logging.get_logger("task_generator.launch").warning(f"planner:={planner_val!r} resolves to mobile:={resolved.adapter_kind!r} but mobile:={explicit_mobile!r} is set explicitly; explicit mobile:= wins")
            else:
                mobile_val = resolved.adapter_kind
                _planner_selector_override = (resolved.selector_key, resolved.selector_value)

        human_launch = IncludeLaunchDescription(
            PathJoinSubstitution(
                [
                    FindPackageShare("task_generator"),
                    "launch",
                    "human",
                    "human.launch.py",
                ]
            ),
            launch_arguments={
                "simulator": human_val,
                "namespace": allocated_ns,
                # Launch substitutions preserve a relative value as relative
                # to each node namespace.  Keep this explicitly absolute so
                # auditory nodes do not resolve it below task_generator_node.
                "environment_namespace": (
                    "/" + os.path.dirname(allocated_ns).strip("/")
                ),
                **auditory.dict,
                **auditory_viz.dict,
                **auditory_playback.dict,
                **auditory_block_size.dict,
                **auditory_assets.dict,
                **auditory_sound_dir.dict,
                **auditory_propagation.dict,
                **auditory_multi_portal.dict,
                **auditory_rir_in_propagation.dict,
                **auditory_robot_sound.dict,
                **auditory_motor.dict,
                **auditory_motor_playback.dict,
                **auditory_environment_playback.dict,
                **auditory_listener.dict,
                **auditory_microphones.dict,
                **auditory_viewport_height.dict,
            }.items(),
        )

        pedestrian_marker_node = launch_ros.actions.Node(
            package="rviz_utils",
            executable="hri_producer",
            name="hri_producer",
            namespace=os.path.dirname(allocated_ns),
            parameters=[
                {"use_sim_time": True},
                {"max_bodies": 32},
            ],
            output="screen",
        )

        dotted_overrides: dict[str, object] = {}
        for k, v in context.launch_configurations.items():
            if k.startswith("task."):
                dotted_overrides[k] = launch_str_to_value(v)
            elif k.startswith("mobile.") or k.startswith("arm."):
                # `<cap>.<key>:=<val>` becomes `robot.<cap>.<key>` and lands as a
                # kwarg in RobotManager._adapter_kwargs_for, overlaying the
                # cap-file YAML for the bound adapter.
                dotted_overrides[f"robot.{k}"] = launch_str_to_value(v)
        if _planner_selector_override is not None:
            sel_key, sel_val = _planner_selector_override
            param_key = f"robot.mobile.{sel_key}"
            if param_key not in dotted_overrides:
                dotted_overrides[param_key] = sel_val

        dotted_overrides.update(expand_flag_namespace(context, "optim", launch_str_to_value))
        debug_flags = expand_flag_namespace(context, "debug", launch_str_to_value)
        dotted_overrides.update(debug_flags)

        # launch_ros.normalize_parameters turns list values into tuples and
        # yaml.dump emits them with !!python/tuple, which rcl drops silently.
        # Bypass by writing our own params yaml and passing the path.
        overrides_files: list[str] = []
        if dotted_overrides:
            fh = tempfile.NamedTemporaryFile(mode='w', prefix='arena_overrides_', suffix='.yaml', delete=False)
            yaml.safe_dump(
                {'/**': {'ros__parameters': dotted_overrides}},
                fh,
                default_flow_style=False,
            )
            fh.close()
            overrides_files.append(fh.name)

        task_generator_node = launch_ros.actions.Node(
            package="task_generator",
            executable="task_generator_node",
            namespace=os.path.dirname(allocated_ns),
            name=os.path.basename(allocated_ns),
            output="screen",
            parameters=[
                {
                    "use_sim_time": True,
                    "sim": arena_sim,
                    "human": human_val,
                    "robot.mobile_adapter": mobile_val,
                    "robot.arm_adapter": arm_val,
                    **robot.str_param,
                    **tm_robots.str_param,
                    **tm_obstacles.str_param,
                    "tm_modules": tm_modules_val,
                    **world.str_param,
                    "static_audio_devices": auditory_static_devices.param_value(str),
                    **record_data_dir.str_param,
                    **auto_reset.param(bool),
                    **fail_on_collision.param(bool),
                    **train_mode.param(bool),
                    "env_id": allocated_id,
                    "prefix": prefix_val,
                },
                parameter_file.substitution,
                {
                    **episodes.param(int),
                    'task.scenario.file': scenario_file.substitution,
                },
                *overrides_files,
            ],
        )

        aiomonitor_port = 20101 + max(allocated_id, 0) * 10
        debug_window_cb = launch.event_handlers.OnProcessStart(
            target_action=task_generator_node,
            on_start=[
                ExecuteProcess(
                    cmd=[
                        "/usr/bin/x-terminal-emulator",
                        "-e",
                        f'bash -c "sleep 5; python -m aiomonitor.cli -p {aiomonitor_port}"',
                    ],
                    output="screen",
                )
            ],
        )

        data_recorder_process = launch.actions.ExecuteProcess(
            cmd=[
                'ros2',
                'run',
                'arena_evaluation',
                'record',
                '--ros-args',
                '-p',
                'use_sim_time:=true',
                '-p',
                ['record_data_dir:=', record_data_dir.substitution],
                '-r',
                ['__ns:=/', allocated_ns],
            ],
            output='screen',
            condition=launch.conditions.IfCondition(launch.substitutions.PythonExpression(["'", record_data_dir.substitution, "' != '' and '", disable_auto_recorder.substitution, "' != 'true'"])),
        )

        shutdown_on_node_exit = RegisterEventHandler(
            OnProcessExit(
                target_action=task_generator_node,
                on_exit=[launch.actions.Shutdown(reason="task_generator_node exited")],
            )
        )

        env_actions: list[launch.LaunchDescriptionEntity] = [
            IsolatedGroupAction([human_launch, pedestrian_marker_node, task_generator_node, data_recorder_process]),
        ]
        if truthy(debug_flags.get("debug.aiomonitor")):
            env_actions.append(launch.actions.RegisterEventHandler(debug_window_cb))
        env_actions.append(shutdown_on_node_exit)
        return env_actions

    def _make_env(context: launch.LaunchContext) -> list[launch.LaunchDescriptionEntity]:
        managed_val = launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(managed.substitution)).lower() in ("true", "1")
        sim_val = launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(sim.substitution))
        if managed_val:
            if not sim_val:
                raise RuntimeError("managed:=true requires sim:= (arena passes it automatically)")
            allocated_id = int(launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(env_id.substitution)))
            allocated_ns = launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(ns.substitution)).lstrip("/")
            arena_sim = sim_val
        else:
            requested = int(launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(env_id.substitution)))
            ns_val = launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(ns.substitution))
            allocated_id, allocated_ns, arena_sim = _allocate_env(requested, ns_val)
            if sim_val and sim_val != arena_sim:
                raise RuntimeError(f"sim:={sim_val} requested but the arena runtime is running {arena_sim}; shut down the runtime or omit sim:=")
        return _build_env_actions(allocated_id, allocated_ns, arena_sim, context)

    return launch.LaunchDescription(
        [
            *ld_items,
            SetGlobalLogLevelAction(log_level.substitution),
            OpaqueFunction(function=_make_env),
        ]
    )


if __name__ == "__main__":
    generate_launch_description()
