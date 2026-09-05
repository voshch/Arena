import itertools
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from arena_bringup.future import IfElseSubstitution, PythonExpression  # noqa
from arena_bringup.substitutions import LaunchArgument

# Appended to gz's default gui.config so the ViewportCamera GUI plugin loads
# alongside the stock GUI (see _render_gui_config).
_VIEWPORT_GUI_PLUGIN = """\
<plugin filename="ViewportCamera" name="Arena viewport camera">
  <gz-gui>
    <title>Arena viewport camera</title>
    <property type="bool" key="showTitleBar">true</property>
    <property type="string" key="state">docked</property>
  </gz-gui>
</plugin>
"""


def _prune_dead_ament_prefixes() -> None:
    """Drop AMENT_PREFIX_PATH entries whose package-index markers all dangle."""
    prefixes = [p for p in os.environ.get('AMENT_PREFIX_PATH', '').split(os.pathsep) if p]
    kept: list[str] = []

    for prefix in prefixes:
        index = os.path.join(prefix, 'share', 'ament_index', 'resource_index', 'packages')
        try:
            entries = os.listdir(index)
        except OSError:
            kept.append(prefix)
            continue
        if not entries or any(os.path.isfile(os.path.join(index, name)) for name in entries):
            kept.append(prefix)

    if len(kept) != len(prefixes):
        os.environ['AMENT_PREFIX_PATH'] = os.pathsep.join(kept)


def _select_render_engine() -> str:
    """ogre2 renders PBR materials but needs a real GL device; ogre1 is the software-safe fallback.
    Forced-software GL (LIBGL_ALWAYS_SOFTWARE) segfaults ogre2's GL3Plus backend, so it pins ogre1."""
    if os.environ.get("LIBGL_ALWAYS_SOFTWARE", "0").lower() in ("1", "true"):
        return "ogre"
    try:
        subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
        return "ogre"
    return "ogre2"


def generate_launch_description():

    _prune_dead_ament_prefixes()

    use_sim_time = LaunchArgument(
        "use_sim_time",
        default_value='True',
        description="Use simulation (Gazebo) clock if true",
    )

    world = LaunchArgument(
        "world",
        default_value='',
        description="World name",
    )

    headless = LaunchArgument(
        'headless',
    )

    # Injects PedSkeletonPlugin into the world SDF so gpu_lidar perceives
    # articulated pedestrians. ped_skeleton_enabled:=false to disable.
    ped_skeleton_enabled = LaunchArgument(
        'ped_skeleton_enabled',
        default_value='true',
        description='Enable pedestrian skeletal articulation plugin (PedSkeletonPlugin)',
    )

    # Set environment variables
    package_root = get_package_share_directory('arena_bringup')
    ss_root = get_package_share_directory('arena_simulation_setup')
    robots_root = get_package_share_directory('arena_robots')

    # Set paths for Gazebo, Physics Engine, and Resource

    # GZ_CONFIG_PATHS = [
    #     # os.path.join(get_package_share_directory('gz-sim8'), "gz"),
    #     # os.path.join(workspace_root, 'install', 'gz-tools2', 'share', 'gz'),
    # ]

    # GZ_SIM_PHYSICS_ENGINE_PATH = os.path.join(
    #     workspace_root, "build", "gz-physics7"
    # )

    staging_path = os.path.join(package_root, '..', 'staging')
    os.makedirs(staging_path, exist_ok=True)

    from arena_simulation_setup.staging import stage
    try:
        stage(staging_path)
    except Exception as exc:
        print(f'[gazebo.launch] model staging failed: {exc}', file=sys.stderr)

    GZ_SIM_RESOURCE_PATHS = [
        os.path.join(staging_path),
        robots_root,
        os.path.dirname(robots_root),
        os.path.join(ss_root, "assets", "Common", "Human", "arenian", "arenian.sdf"),
        os.path.join(ss_root, "assets", "Common", "Human", "arenian"),
    ]

    deps_file = os.path.join(staging_path, 'deps')
    if os.path.isfile(deps_file):
        with open(deps_file) as f:
            deps = f.readlines()
            for package in itertools.chain(deps, ('arena_simulation_setup',)):
                try:
                    package_path = get_package_share_directory(package.strip())
                    GZ_SIM_RESOURCE_PATHS.append(os.path.join(package_path, '..'))
                except BaseException:
                    pass

    GZ_SIM_RESOURCE_PATHS = [os.path.normpath(path) for path in GZ_SIM_RESOURCE_PATHS]

    for root, dirs, files in os.walk(os.path.join(ss_root, "gazebo_models")):
        for dir_name in dirs:
            if 'hospital' in dir_name.lower():
                GZ_SIM_RESOURCE_PATHS.append(os.path.join(root, dir_name))

    GZ_SIM_RESOURCE_PATHS_COMBINED = ":".join(GZ_SIM_RESOURCE_PATHS)

    # Update environment variables
    model_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    if model_path:
        GZ_SIM_RESOURCE_PATHS_COMBINED = f"{model_path}:{GZ_SIM_RESOURCE_PATHS_COMBINED}"
    os.environ["GZ_SIM_RESOURCE_PATH"] = GZ_SIM_RESOURCE_PATHS_COMBINED
    os.environ["GAZEBO_MODEL_PATH"] = GZ_SIM_RESOURCE_PATHS_COMBINED
    # os.environ["GZ_SIM_PHYSICS_ENGINE_PATH"] = GZ_SIM_PHYSICS_ENGINE_PATH

    desired_world = PathJoinSubstitution([
        ss_root,
        "worlds",
        world.substitution,
        "worlds",
        PythonExpression(['"', world.substitution, '.world"']),
    ])

    world_path = IfElseSubstitution(
        condition=PythonExpression(['not os.path.isfile("', desired_world, '")'], python_modules=['os']),
        if_value=PathJoinSubstitution([
            package_root,
            'configs',
            'gazebo',
            'empty.sdf',
        ]),
        else_value=desired_world,
    )

    def _inject_ped_skeleton_plugin(world_sdf_path: str, enabled: bool) -> str:
        """Return a patched world SDF path with PedSkeletonPlugin injected.

        Parses the world SDF, appends the plugin element to the <world> element,
        writes a temp file, and returns its path. When disabled, returns the
        original path unchanged.
        """
        if not enabled:
            return world_sdf_path

        try:
            tree = ET.parse(world_sdf_path)
            root = tree.getroot()
            world_el = root.find('world')
            if world_el is None:
                return world_sdf_path

            plugin_el = ET.SubElement(world_el, 'plugin')
            plugin_el.set('filename', 'PedSkeletonPlugin')
            plugin_el.set('name', 'arena_gz_plugins::PedSkeletonPlugin')
            enabled_el = ET.SubElement(plugin_el, 'enabled')
            enabled_el.text = 'true'

            tmp = tempfile.NamedTemporaryFile(
                mode='wb', suffix='.sdf', delete=False,
                prefix='arena_world_',
            )
            tree.write(tmp, xml_declaration=True, encoding='utf-8')
            tmp.close()
            return tmp.name

        except Exception as exc:
            # Non-fatal: fall back to original world without the plugin.
            print(f'[gazebo.launch] PedSkeletonPlugin injection failed: {exc}', file=sys.stderr)
            return world_sdf_path

    def _render_gui_config(engine: str) -> str:
        """Render a --gui-config temp file from the user's own ~/.gz layout (else the
        stock config) with the engine pinned and the ViewportCamera plugin ensured."""
        user_config = os.path.join(os.path.expanduser('~'), '.gz', 'sim', '8', 'gui.config')
        checked_in_config = os.path.join(package_root, 'configs', 'gazebo', 'gui.config')
        base = user_config if os.path.isfile(user_config) else checked_in_config
        try:
            with open(base) as f:
                content = f.read()
        except Exception as exc:
            print(f'[gazebo.launch] gui.config read failed ({base}): {exc}', file=sys.stderr)
            content = ''
        # The default pins ogre2, so match it to the engine actually selected.
        content = re.sub(r'<engine>[^<]*</engine>', f'<engine>{engine}</engine>', content)
        if 'Arena viewport camera' not in content:
            # gz config is a flat list of top-level elements, so append the plugin verbatim.
            content = content.rstrip() + '\n\n' + _VIEWPORT_GUI_PLUGIN.strip() + '\n'
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.config', delete=False, prefix='arena_gui_',
        )
        tmp.write(content)
        tmp.close()
        return tmp.name

    def _launch_gazebo(context, *args, **kwargs):
        resolved_world = context.perform_substitution(world_path)
        headless_val = context.perform_substitution(headless.substitution)
        skeleton_val = context.perform_substitution(ped_skeleton_enabled.substitution)
        skeleton_on = skeleton_val.lower() not in ('false', '0')
        resolved_world = _inject_ped_skeleton_plugin(resolved_world, skeleton_on)
        engine = _select_render_engine()
        gz_args = resolved_world + f" -r --render-engine {engine}"
        if headless_val.lower() in ("true", "1"):
            gz_args += " -s"
            if engine == "ogre2":
                gz_args += " --headless-rendering"
        else:
            gz_args += f" --gui-config {_render_gui_config(engine)}"
        include = IncludeLaunchDescription(
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py',
            ]),
            launch_arguments={
                "gz_version": "8",
                "gz_args": gz_args,
                "physics-engine": "gz-physics-dartsim",
            }.items(),
        )
        return [LogInfo(msg=f"gazebo render engine: {engine} (ogre2 = PBR/textures, needs GPU)"), include]

    gazebo = OpaqueFunction(function=_launch_gazebo)

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        parameters=[{
            **use_sim_time.dict,
            'config_file': PathJoinSubstitution([
                FindPackageShare('arena_bringup'),
                'configs',
                'gazebo',
                'clock_bridge.yaml',
            ]),
        }],
    )

    clock_relay = Node(
        package='arena_runtime',
        executable='clock_relay',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    # Point gz-sim at <install>/lib so it can load PedSkeletonPlugin.
    try:
        arena_gz_plugins_lib = os.path.join(
            get_package_share_directory('arena_gz_plugins'), '..', '..', 'lib'
        )
        arena_gz_plugins_lib = os.path.normpath(arena_gz_plugins_lib)
    except Exception:
        arena_gz_plugins_lib = ''

    existing_plugin_path = os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')
    gz_plugin_path = (
        f"{arena_gz_plugins_lib}:{existing_plugin_path}"
        if arena_gz_plugins_lib and existing_plugin_path
        else (arena_gz_plugins_lib or existing_plugin_path)
    )

    # The ViewportCamera GUI plugin loads from the same <install>/lib via GZ_GUI_PLUGIN_PATH.
    existing_gui_plugin_path = os.environ.get('GZ_GUI_PLUGIN_PATH', '')
    gz_gui_plugin_path = (
        f"{arena_gz_plugins_lib}:{existing_gui_plugin_path}"
        if arena_gz_plugins_lib and existing_gui_plugin_path
        else (arena_gz_plugins_lib or existing_gui_plugin_path)
    )

    # Return the LaunchDescription with all the nodes/actions

    return LaunchDescription(
        [
            use_sim_time,
            world,
            headless,
            ped_skeleton_enabled,
            # SetEnvironmentVariable(
            #     "GZ_SIM_PHYSICS_ENGINE_PATH", GZ_SIM_PHYSICS_ENGINE_PATH
            # ),
            SetEnvironmentVariable(
                "GZ_SIM_RESOURCE_PATH", GZ_SIM_RESOURCE_PATHS_COMBINED
            ),
            SetEnvironmentVariable(
                "GZ_SIM_SYSTEM_PLUGIN_PATH", gz_plugin_path
            ),
            SetEnvironmentVariable(
                "GZ_GUI_PLUGIN_PATH", gz_gui_plugin_path
            ),
            gazebo,
            # robot_state_publisher,
            # joint_state_publisher,
            # spawn_robot,
            # IncludeLaunchDescription(
            #     PythonLaunchDescriptionSource(random_spawn_launch_file),
            #     condition=IfCondition(random_spawn_test),
            # ),
            clock_bridge,
            clock_relay,
        ]
    )


if __name__ == "__main__":
    generate_launch_description()
