import atexit
import contextlib
import os
import tempfile
import time

import launch
import launch.event_handlers
import launch.launch_description_sources
import launch.substitutions
import launch_ros.actions
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from arena_bringup.actions import IsolatedGroupAction, IsolatedIncludeLaunchDescription
from arena_bringup.defaults import default_human
from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction
from arena_bringup.substitutions import LaunchArgument, deprecated_launch_args
from arena_rclpy_mixins import launch_str_to_value
from launch.actions import (
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from task_generator.constants.runtime import EPISODE_PARAMS
from task_generator.utils.flags import expand_flag_namespace, truthy

_REGISTER_RETRY_SEC = 1.0
_REGISTER_LOG_INTERVAL_SEC = 10.0
_AUTO_ENV_ID = 0xFFFF
_HEARING_PREFIX = "robot.hearing."


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


def generate_launch_description() -> launch.LaunchDescription:
    bringup_dir = get_package_share_directory("arena_bringup")

    ld_items = []
    LaunchArgument.auto_append(ld_items)

    log_level = LaunchArgument(
        name='log_level',
        default_value='warn',
        description='Per-node log level. See launch README.',
    )

    env_id = LaunchArgument(
        name="env.id",
        default_value=str(_AUTO_ENV_ID),
        description="Requested env id; 65535 = auto-allocate via /arena/register_env.",
    )

    managed = LaunchArgument(
        name="env.managed",
        default_value="false",
        description="true = arena pre-reserved; skip /arena/register_env and use env.id/env.ns from args. Placement comes via confirm_world either way.",
    )

    ns = LaunchArgument(
        name="env.ns",
        default_value="",
        description="Explicit ns path (e.g. for sim2real); empty = auto-generate.",
    )

    sim = LaunchArgument(
        name="sim",
        default_value="",
        description="empty = adopt the runtime's sim; explicit [dummy, gazebo, isaac] must match the runtime",
    )
    env_tf = LaunchArgument(
        name="env.tf",
        choices=["auto", "env", "global"],
        default_value="auto",
        description="tf topics: env = <env ns>/tf and <env ns>/tf_static, global = /tf and /tf_static, auto = global for sim isaac or robot.train, env otherwise.",
    )
    # human/mobile defaults derive from arena's authoritative `sim` (the RegisterEnv
    # response, or the sim arg arena passes for managed envs). Empty here means
    # "use arena_sim". User can still override by passing e.g. human:=dummy explicitly.
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
    for name, description in (
        ("auditory.viz.enabled", "Publish source, portal and listener propagation markers."),
        ("auditory.output.device", "PortAudio output device for workstation playback. auto tries pulse, pipewire, default, then the PortAudio default. none starts no listener renderer."),
        ("auditory.output.block_size", "Workstation audio callback block size."),
        ("auditory.output.buffer_s", "Workstation jitter buffer target in seconds, raise it on repeated underflows."),
        ("auditory.output.motor.enabled", "Play robot motor audio on the workstation."),
        ("auditory.output.ambient.enabled", "Play environment audio on the workstation. Emission and robot hearing continue when false."),
        ("auditory.propagation.backend", "Propagation backend, pyroomacoustics, level3 or legacy."),
        ("auditory.portal.multi_hop.enabled", "Allow pyroomacoustics RIR rendering across multi-hop door and opening portal routes."),
        ("auditory.rir.max_order", "Image-source reflection order of every RIR."),
        ("auditory.pedestrian_listeners.enabled", "Pedestrians are propagation listeners and receive sound stimuli through the human simulator."),
        ("auditory.motor.enabled", "Let robots emit drivetrain audio. Robots stay listeners regardless."),
        ("auditory.motor.trim_db", "Live offset in dB on the motor asset level, lower it to attenuate ego-noise."),
        ("auditory.listener.id", "Microphone listener id that feeds the listener renderer, the RViz auditory panel switches it at run time."),
        ("auditory.viewport.height_m", "Listening height of the viewport camera's down-projection microphone."),
        ("auditory.array.spec", "Robot microphone array, stereo, four_mic, mono or a yaml path. Empty is four_mic when robot.hearing is srp or seld."),
        ("auditory.array.mount_frame", "TF frame the robot microphone array is mounted on, {prefix} and {base_frame} expand, a bare leaf joins the robot prefix. Empty uses the robot base frame."),
        ("robot.hearing.seld.device", "Torch device of the SELDnet front-end."),
        ("robot.hearing.seld.lookahead_frames", "SELDnet front-end label frames of future context, 100 ms each."),
        ("robot.hearing.seld.bearing_source", "SELDnet front-end bearing, gcc fits GCC-PHAT over the array, seld takes the model azimuth."),
        ("robot.hearing.srp.hop_s", "srp front-end hop length in seconds."),
        ("robot.hearing.srp.floor_window_s", "srp front-end noise-floor median window in seconds."),
        ("robot.hearing.srp.onset_db", "srp front-end onset threshold above the floor in dB."),
    ):
        LaunchArgument(name=name, default_value="", description=f"{description} Empty = node default.")
    LaunchArgument(
        name="auditory.motor.model",
        choices=["", "procedural", "wav"],
        default_value="",
        description="Robot motor audio, calibrated procedural synthesis (Jackal, other models use WAVs) or WAV loops. Empty = node default.",
    )
    auditory_static_sounds = LaunchArgument(
        name="auditory.static_sounds",
        default_value="[]",
        description="YAML list of world-independent sound entities (radios, alarms), same Sound schema as world.yaml sounds. Non-empty enables the sounds module even with auditory:=none.",
    )
    LaunchArgument(
        name="auditory.microphones",
        default_value="[]",
        description="YAML list of robot microphone mappings (owner, robot, placement, frame, index).",
    )
    robot = LaunchArgument(name="robot", default_value="auto")
    tm_robots = LaunchArgument(name="task.robots", default_value="explore")
    tm_config = LaunchArgument(
        name="task.config",
        default_value="",
        description="Task config file (task_modes list). Overrides task.robots. Bare names resolve under arena_bringup/configs/tasks.",
    )
    for name, description in EPISODE_PARAMS.items():
        LaunchArgument(name=name, default_value='', description=f'{description} Empty = node default.')
    scenario_file = LaunchArgument(
        name='task.scenario.file',
        default_value='',
        description='Scenario for the scenario task modes (empty = use task.params default).',
    )
    scenario_linger = LaunchArgument(
        name='task.scenario.linger_after_completion',
        default_value='false',
        description='Keep a completed scenario robot task alive until timeout/external cancellation.',
    )
    tm_obstacles = LaunchArgument(name="task.obstacles", default_value="random")
    tm_modules = LaunchArgument(name="task.modules", default_value="rviz_ui")
    LaunchArgument(name="optim", default_value=os.environ.get("ARENA_OPTIM", ""))
    world = LaunchArgument(name="world", default_value="map_empty")
    mobile = LaunchArgument(
        name="robot.mobile",
        default_value="",
        description="mobile adapter kind; empty = derive from arena_sim ({dummy: none, *: nav2})",
    )
    planner = LaunchArgument(
        name="robot.planner",
        default_value="",
        description="top-level planner selector; resolves to robot.mobile:=<adapter> robot.mobile.<selector>:=<name> via arena_planners.resolver",
    )
    hearing = LaunchArgument(
        name="robot.hearing",
        choices=["none", "bus", "srp", "seld"],
        default_value="none",
        description="Robot-side hearing layer (arena_auditory.hearing) of every fleet robot: belief grid, a Nav2 speed-filter mask merged into each robot's nav2 params, RViz displays. Event source: the simulator bus, the untrained onset + GCC-PHAT front-end on any array, or the live SELDnet front-end on the array its weights were trained on. Needs auditory:=arena.",
    )
    hearing_policy = LaunchArgument(
        name="robot.hearing.policy",
        choices=["belief", "listen", "full"],
        default_value="full",
        description="Hearing mask layers: belief only, plus the corner listen cap, plus yield.",
    )
    arm = LaunchArgument(
        name="robot.arm",
        default_value="moveit",
        description="arm adapter kind",
    )
    record_dir = LaunchArgument(
        name="record.dir",
        default_value="",
        description="Directory for episode recording; empty disables.",
    )
    record_auto = LaunchArgument(
        name="record.auto",
        default_value="true",
        description="Spawn the data recorder when record.dir is set.",
    )
    LaunchArgument(
        name="debug",
        default_value="",
        description="comma list of debug tokens (e.g. aiomonitor,map_server); also debug.<token>:=true",
    )
    train_mode = LaunchArgument(name="robot.train", default_value="false")
    parameter_file = LaunchArgument(
        name="task.params",
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

        env_tf_val = launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(env_tf.substitution))
        if env_tf_val == "auto":
            train_val = launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(train_mode.substitution))
            env_tf_val = "global" if arena_sim == "isaac" or truthy(train_val) else "env"
        if env_tf_val == "env" and arena_sim == "isaac":
            raise RuntimeError("env.tf:=env is not supported with sim isaac, its robot odom tf is published on /tf")
        env_ns = os.path.dirname(allocated_ns).strip("/")
        tf_namespace = f"/{env_ns}" if env_tf_val == "env" and env_ns else ""
        tf_remaps = [launch_ros.actions.SetRemap(topic, tf_namespace + topic) for topic in ("/tf", "/tf_static")] if tf_namespace else []

        human_val = launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(human.substitution)) or default_human(arena_sim)
        auditory_val = launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(auditory.substitution))
        hearing_val = launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(hearing.substitution))
        hearing_policy_val = launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(hearing_policy.substitution))
        if hearing_val != "none" and auditory_val == "none":
            raise RuntimeError(f"robot.hearing:={hearing_val} needs auditory:=arena")
        mobile_val = launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(mobile.substitution)) or {"dummy": "none"}.get(arena_sim, "nav2")
        arm_val = launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(arm.substitution))
        tm_modules_val = launch.utilities.perform_substitutions(
            context,
            launch.utilities.normalize_to_list_of_substitutions(tm_modules.substitution),
        )
        configured_modules = [
            value.strip()
            for value in tm_modules_val.split(",")
            if value.strip()
        ]
        static_sounds_val = launch.utilities.perform_substitutions(
            context,
            launch.utilities.normalize_to_list_of_substitutions(
                auditory_static_sounds.substitution
            ),
        ).strip()
        sounds_enabled = auditory_val != "none" or static_sounds_val not in ("", "[]")
        if sounds_enabled and "sounds" not in configured_modules:
            configured_modules.append("sounds")
        tm_modules_val = ",".join(configured_modules)
        if "sounds" in configured_modules:
            try:
                get_package_share_directory("arena_auditory")
            except PackageNotFoundError as exc:
                raise RuntimeError("the sounds module (auditory:=arena, robot.hearing, auditory.static_sounds or task.modules:=sounds) needs the arena_auditory package, install it with `arena feature auditory install`") from exc

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
                launch.logging.get_logger("task_generator.launch").warning(f"robot.planner:={planner_val!r} resolves to robot.mobile:={resolved.adapter_kind!r} but robot.mobile:={explicit_mobile!r} is set explicitly; explicit robot.mobile:= wins")
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
            }.items(),
        )

        auditory_actions: list[launch.LaunchDescriptionEntity] = []
        if auditory_val != "none":
            auditory_actions.append(
                launch.actions.GroupAction(
                    [
                        IncludeLaunchDescription(
                            PathJoinSubstitution(
                                [
                                    FindPackageShare("task_generator"),
                                    "launch",
                                    "auditory",
                                    "auditory.launch.py",
                                ]
                            ),
                            launch_arguments={
                                "simulator": auditory_val,
                                "namespace": allocated_ns,
                                "environment_namespace": ("/" + os.path.dirname(allocated_ns).strip("/")),
                            }.items(),
                        ),
                    ]
                )
            )

        hearing_actions: list[launch.LaunchDescriptionEntity] = []
        if hearing_val != "none":
            hearing_overrides = {key: value for key, value in context.launch_configurations.items() if key.startswith(_HEARING_PREFIX) and key != hearing_policy.name and value}
            hearing_actions.append(
                IsolatedIncludeLaunchDescription(
                    launch.launch_description_sources.PythonLaunchDescriptionSource(
                        os.path.join(get_package_share_directory("arena_auditory"), "launch", "hearing.launch.py"),
                    ),
                    args={
                        **hearing_overrides,
                        "env.ns": "/" + os.path.dirname(allocated_ns).strip("/"),
                        "tg_node": os.path.basename(allocated_ns),
                        "frontend": hearing_val,
                        "policy": hearing_policy_val,
                    },
                )
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

        declared = {a.name for a in ld_items}
        dotted_overrides: dict[str, object] = {}
        for k, v in context.launch_configurations.items():
            if k in EPISODE_PARAMS:
                if v:
                    dotted_overrides[k] = yaml.safe_load(v)
                continue
            if k in declared:
                continue
            if k.startswith(("task.", "robot.")) and not k.startswith(_HEARING_PREFIX):
                # `robot.<cap>.<key>:=<val>` lands as a kwarg in
                # RobotManager._adapter_kwargs_for, overlaying the cap-file
                # YAML for the bound adapter.
                dotted_overrides[k] = launch_str_to_value(v)
        if hearing_val != "none" and "robot.mobile.params_overlay" not in dotted_overrides:
            dotted_overrides["robot.mobile.params_overlay"] = os.path.join(get_package_share_directory("arena_auditory"), "config", "hearing", "nav2_overlay.yaml")
        if _planner_selector_override is not None:
            sel_key, sel_val = _planner_selector_override
            param_key = f"robot.mobile.{sel_key}"
            if param_key not in dotted_overrides:
                dotted_overrides[param_key] = sel_val

        scenario_val = launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(scenario_file.substitution)).strip()
        if scenario_val:
            dotted_overrides["task.scenario.file"] = scenario_val

        tm_config_val = launch.utilities.perform_substitutions(context, launch.utilities.normalize_to_list_of_substitutions(tm_config.substitution)).strip()
        if tm_config_val:
            if os.sep in tm_config_val:
                tm_config_val = os.path.abspath(tm_config_val)
            else:
                if not tm_config_val.endswith((".yaml", ".yml")):
                    tm_config_val += ".yaml"
                tm_config_val = os.path.join(bringup_dir, "configs", "tasks", tm_config_val)
            if not os.path.isfile(tm_config_val):
                raise FileNotFoundError(f"task.config: {tm_config_val} does not exist")

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
                    "auditory": auditory_val,
                    "robot.mobile_adapter": mobile_val,
                    "robot.hearing": hearing_val,
                    "robot.arm_adapter": arm_val,
                    **robot.str_param,
                    "task.robots": tm_robots.param_value(str),
                    "task.obstacles": tm_obstacles.param_value(str),
                    "task.config": tm_config_val,
                    "task.modules": tm_modules_val,
                    **world.str_param,
                    "auditory.static_sounds": auditory_static_sounds.param_value(str),
                    "robot.train": train_mode.param_value(bool),
                    "env_id": allocated_id,
                    "prefix": prefix_val,
                    "tf_namespace": tf_namespace,
                },
                parameter_file.substitution,
                {
                    "task.scenario.linger_after_completion": scenario_linger.param_value(bool),
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
                ['record_data_dir:=', record_dir.substitution],
                '-r',
                ['__ns:=/', allocated_ns],
                *(arg for topic in ("/tf", "/tf_static") if tf_namespace for arg in ('-r', f'{topic}:={tf_namespace}{topic}')),
            ],
            output='screen',
            condition=launch.conditions.IfCondition(launch.substitutions.PythonExpression(["'", record_dir.substitution, "' != '' and '", record_auto.substitution, "'.lower() in ('true', '1')"])),
        )

        shutdown_on_node_exit = RegisterEventHandler(
            OnProcessExit(
                target_action=task_generator_node,
                on_exit=[launch.actions.Shutdown(reason="task_generator_node exited")],
            )
        )

        env_actions: list[launch.LaunchDescriptionEntity] = [
            IsolatedGroupAction([*tf_remaps, human_launch, *auditory_actions, *hearing_actions, pedestrian_marker_node, task_generator_node, data_recorder_process]),
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
                raise RuntimeError("env.managed:=true requires sim:= (arena passes it automatically)")
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

    def _aliases(context: launch.LaunchContext) -> None:
        deprecated_launch_args(context)

    return launch.LaunchDescription(
        [
            OpaqueFunction(function=_aliases),
            *ld_items,
            SetGlobalLogLevelAction(log_level.substitution),
            OpaqueFunction(function=_make_env),
        ]
    )


if __name__ == "__main__":
    generate_launch_description()
