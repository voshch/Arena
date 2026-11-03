"""Field-level schema validation for new arena_runtime_msgs types.

Instantiates each type, populates fields, reads them back, and asserts
equality. No ROS runtime required, this validates that rosidl-generated
types expose the correct field names and accept the right Python values.
Run after `colcon build` installs the generated Python bindings.
"""


def test_episode_record_fields():
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from diagnostic_msgs.msg import KeyValue
    from arena_runtime_msgs.msg import EpisodeRecord

    r = EpisodeRecord()
    r.episode_id = 7
    r.world = "map1"
    r.seed = 12345
    r.modes = [KeyValue(key="robots", value="random"), KeyValue(key="modules", value="benchmark,staged")]
    r.robots = ["husky", "jackal"]
    r.outcome_state = EpisodeRecord.SUCCESS
    r.outcome_info = "reached goal"
    r.goal_uuid = "abc-123"
    r.integrity = ["env_0/jackal moved by hand"]

    p = Parameter()
    p.name = "static.n"
    p.value = ParameterValue()
    p.value.type = ParameterType.PARAMETER_INTEGER
    p.value.integer_value = 4
    p.name = "task.random.static.n"
    r.params = [p]
    r.human_params = [p]

    assert r.episode_id == 7
    assert r.world == "map1"
    assert r.seed == 12345
    assert [(m.key, m.value) for m in r.modes] == [("robots", "random"), ("modules", "benchmark,staged")]
    assert list(r.robots) == ["husky", "jackal"]
    assert r.outcome_state == EpisodeRecord.SUCCESS
    assert r.outcome_info == "reached goal"
    assert r.goal_uuid == "abc-123"
    assert list(r.integrity) == ["env_0/jackal moved by hand"]
    assert len(r.params) == 1
    assert r.params[0].name == "task.random.static.n"
    assert r.params[0].value.integer_value == 4
    assert [q.name for q in r.human_params] == ["static.n"]

    assert EpisodeRecord.QUEUED == 0
    assert EpisodeRecord.RUNNING == 1
    assert EpisodeRecord.SUCCESS == 2
    assert EpisodeRecord.FAILED == 3
    assert EpisodeRecord.SKIPPED == 4
    assert EpisodeRecord.FATAL == 5


def test_episode_outcome_constants_match_the_run_episode_result():
    from arena_runtime_msgs.action import RunEpisode
    from arena_runtime_msgs.msg import EpisodeRecord

    def constants(cls: type) -> dict[str, int]:
        return {name: getattr(cls, name) for name in dir(cls) if name.isupper() and isinstance(getattr(cls, name), int)}

    record = constants(EpisodeRecord)
    assert record == constants(RunEpisode.Result)
    assert set(record) >= {"QUEUED", "RUNNING", "SUCCESS", "FAILED", "SKIPPED", "FATAL"}
    assert RunEpisode.Feedback.STARTED == EpisodeRecord.RUNNING


def test_reset_episode_srv():
    from arena_runtime_msgs.srv import ResetEpisode

    req = ResetEpisode.Request()
    req.world = "maze"
    req.seed = 42

    assert req.world == "maze"
    assert req.seed == 42

    res = ResetEpisode.Response()
    res.success = True
    res.error_msg = ""

    assert res.success is True
    assert res.error_msg == ""


def test_query_worlds_srv():
    from arena_runtime_msgs.srv import QueryWorlds

    req = QueryWorlds.Request()
    res = QueryWorlds.Response()
    res.ids = ["map1", "map2"]

    assert res.ids == ["map1", "map2"]
    assert req is not None


def test_query_scenarios_srv():
    from arena_runtime_msgs.srv import QueryScenarios

    req = QueryScenarios.Request()
    req.world = "maze"
    res = QueryScenarios.Response()
    res.ids = ["scenario_a"]

    assert req.world == "maze"
    assert res.ids == ["scenario_a"]


def test_query_robots_srv():
    from arena_runtime_msgs.srv import QueryRobots

    res = QueryRobots.Response()
    res.ids = ["turtlebot3"]
    assert res.ids == ["turtlebot3"]


def test_query_static_obstacles_srv():
    from arena_runtime_msgs.srv import QueryStaticObstacles

    res = QueryStaticObstacles.Response()
    res.ids = ["box", "barrel"]
    assert res.ids == ["box", "barrel"]


def test_query_dynamic_obstacles_srv():
    from arena_runtime_msgs.srv import QueryDynamicObstacles

    res = QueryDynamicObstacles.Response()
    res.ids = ["pedestrian_adult"]
    assert res.ids == ["pedestrian_adult"]


def test_query_environments_srv():
    from arena_runtime_msgs.srv import QueryEnvironments

    res = QueryEnvironments.Response()
    res.ids = ["env_a"]
    assert res.ids == ["env_a"]


def test_query_parametrizeds_srv():
    from arena_runtime_msgs.srv import QueryParametrizeds

    res = QueryParametrizeds.Response()
    res.ids = ["param_set_1"]
    assert res.ids == ["param_set_1"]


def test_queue_episode_srv():
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from diagnostic_msgs.msg import KeyValue
    from arena_runtime_msgs.srv import QueueEpisode

    req = QueueEpisode.Request()
    req.action = QueueEpisode.Request.MERGE
    req.modes = [KeyValue(key="obstacles", value="parametrized"), KeyValue(key="modules", value="")]
    req.world = "maze"

    p = Parameter()
    p.name = "static.n"
    p.value = ParameterValue()
    p.value.type = ParameterType.PARAMETER_INTEGER
    p.value.integer_value = 5
    req.params = [p]
    req.human_params = [p]

    assert req.action == QueueEpisode.Request.MERGE
    assert QueueEpisode.Request.MERGE == 0
    assert [(m.key, m.value) for m in req.modes] == [("obstacles", "parametrized"), ("modules", "")]
    assert req.world == "maze"
    assert len(req.params) == 1
    assert req.params[0].name == "static.n"
    assert [q.name for q in req.human_params] == ["static.n"]

    res = QueueEpisode.Response()
    res.success = True
    res.error_msg = ""
    assert res.success is True


def test_get_task_modes_srv():
    from diagnostic_msgs.msg import KeyValue
    from arena_runtime_msgs.srv import GetTaskModes

    res = GetTaskModes.Response()
    res.modes = [KeyValue(key="robots", value="random")]

    assert [(m.key, m.value) for m in res.modes] == [("robots", "random")]


def test_spawn_static_srv():
    from geometry_msgs.msg import PoseStamped
    from arena_runtime_msgs.srv import SpawnStatic

    req = SpawnStatic.Request()
    req.model = "barrel"
    req.pose = PoseStamped()
    req.use_pose = True

    assert req.model == "barrel"
    assert req.use_pose is True

    res = SpawnStatic.Response()
    res.id = "barrel_0"
    res.success = True
    res.error_msg = ""

    assert res.id == "barrel_0"
    assert res.success is True


def test_spawn_dynamic_srv():
    from geometry_msgs.msg import PoseStamped
    from arena_runtime_msgs.srv import SpawnDynamic

    req = SpawnDynamic.Request()
    req.model = "pedestrian_adult"
    req.pose = PoseStamped()
    req.use_pose = False

    assert req.model == "pedestrian_adult"
    assert req.use_pose is False

    res = SpawnDynamic.Response()
    res.id = "ped_0"
    res.success = True
    res.error_msg = ""

    assert res.id == "ped_0"


def test_spawn_robot_srv():
    from diagnostic_msgs.msg import KeyValue
    from geometry_msgs.msg import PoseStamped
    from arena_runtime_msgs.srv import SpawnRobot

    req = SpawnRobot.Request()
    req.model = "turtlebot3"
    req.name = ""
    req.pose = PoseStamped()
    req.use_pose = False
    req.immediate = True
    req.args = [
        KeyValue(key="mobile.local_planner", value="teb"),
        KeyValue(key="mobile.agent", value=""),
    ]

    assert req.model == "turtlebot3"
    assert req.name == ""
    assert req.immediate is True
    assert len(req.args) == 2
    assert req.args[0].key == "mobile.local_planner"
    assert req.args[0].value == "teb"

    res = SpawnRobot.Response()
    res.name = "turtlebot3_0"
    res.success = True
    res.error_msg = ""

    assert res.name == "turtlebot3_0"
    assert res.success is True


def test_despawn_robot_srv():
    from arena_runtime_msgs.srv import DespawnRobot

    req = DespawnRobot.Request()
    req.name = "turtlebot3_0"

    assert req.name == "turtlebot3_0"

    res = DespawnRobot.Response()
    res.success = True
    res.error_msg = ""

    assert res.success is True


def test_robot_queue_msg():
    from arena_runtime_msgs.msg import RobotDescriptor, RobotQueue

    msg = RobotQueue()
    msg.spawn = [RobotDescriptor(name="jackal_1", model="jackal")]
    msg.despawn = [RobotDescriptor(name="jackal_0", model="jackal")]

    assert len(msg.spawn) == 1
    assert msg.spawn[0].name == "jackal_1"
    assert msg.spawn[0].model == "jackal"
    assert len(msg.despawn) == 1
    assert msg.despawn[0].name == "jackal_0"


def test_robot_fleet_msg():
    from diagnostic_msgs.msg import KeyValue
    from arena_runtime_msgs.msg import RobotCap, RobotDescriptor, RobotFleet, RobotState

    state = RobotState(
        descriptor=RobotDescriptor(name="jackal_0", model="jackal", ns="/jackal_0", frame="jackal_0"),
        caps=[RobotCap(cap="mobile", adapter="nav2", instance="", variant="")],
        params=[KeyValue(key="lidar", value="['sick']")],
    )
    msg = RobotFleet(robots=[state])

    assert len(msg.robots) == 1
    assert msg.robots[0].descriptor.name == "jackal_0"
    assert msg.robots[0].descriptor.model == "jackal"
    assert len(msg.robots[0].caps) == 1
    assert msg.robots[0].caps[0].cap == "mobile"
    assert msg.robots[0].caps[0].adapter == "nav2"
    assert len(msg.robots[0].params) == 1
    assert msg.robots[0].params[0].key == "lidar"


def test_run_episode_action_goal():
    from arena_runtime_msgs.action import RunEpisode

    goal = RunEpisode.Goal()
    goal.world = "maze"
    assert goal.world == "maze"


def test_run_episode_action_result():
    from arena_runtime_msgs.action import RunEpisode

    result = RunEpisode.Result()
    result.state = RunEpisode.Result.SUCCESS
    result.info = ""
    result.episode_id = 3

    assert result.state == RunEpisode.Result.SUCCESS
    assert result.episode_id == 3

    assert RunEpisode.Result.QUEUED == 0
    assert RunEpisode.Result.RUNNING == 1
    assert RunEpisode.Result.SUCCESS == 2
    assert RunEpisode.Result.FAILED == 3
    assert RunEpisode.Result.SKIPPED == 4
    assert RunEpisode.Result.FATAL == 5


def test_run_episode_action_feedback():
    from arena_runtime_msgs.action import RunEpisode

    fb = RunEpisode.Feedback()
    fb.state = RunEpisode.Feedback.STARTED
    assert fb.state == RunEpisode.Feedback.STARTED
    assert RunEpisode.Feedback.STARTED == 1


def test_spawn_sound_srv():
    from arena_runtime_msgs.srv import SpawnSound

    req = SpawnSound.Request()
    req.pose.header.frame_id = "map"
    req.kind = "music"
    req.customize_playback = True
    req.asset_id = "custom_radio"
    req.source_volume_db = 70.0
    req.loop = True
    req.initially_active = False
    assert req.pose.header.frame_id == "map"
    assert req.kind == "music"
    assert req.asset_id == "custom_radio"
    assert req.initially_active is False

    res = SpawnSound.Response()
    res.entity = "runtime_music_1"
    res.success = True
    assert res.entity == "runtime_music_1"
    assert res.success is True


def test_remove_sound_srv():
    from arena_runtime_msgs.srv import RemoveSound

    req = RemoveSound.Request()
    req.entity = "runtime_music_1"
    res = RemoveSound.Response()
    res.success = True

    assert req.entity == "runtime_music_1"
    assert res.success is True


def test_adapter_viz_manifest_plugins():
    from arena_runtime_msgs.msg import AdapterPlugin, AdapterVizManifest

    panel = AdapterPlugin(role="panel", class_name="arena_auditory_viz::AuditoryPanel", name="Auditory", properties_json='{"Node": "/env_0/arena_env_node"}')
    tool = AdapterPlugin(role="tool", class_name="arena_auditory_viz::SoundTool", properties_json="{}")
    manifest = AdapterVizManifest(plugins=[panel, tool])

    assert [plugin.role for plugin in manifest.plugins] == ["panel", "tool"]
    assert manifest.plugins[0].name == "Auditory"
    assert manifest.plugins[1].name == ""
    assert list(manifest.env_displays) == []
    assert list(manifest.entries) == []


def test_recorded_topics_msg():
    from arena_runtime_msgs.msg import RecordedTopic, RecordedTopics

    row = RecordedTopic(
        key="sound_events",
        topic="{tg}/sound_events",
        msg_type="arena_auditory_msgs/msg/SoundEvent",
        robot_scoped=False,
        throttled=False,
        qos_transient_local=False,
        recorded=True,
    )
    robot_row = RecordedTopic(key="heard_sound", topic="{ns}/heard_sound", msg_type="arena_auditory_msgs/msg/HeardSoundEvent", robot_scoped=True)
    msg = RecordedTopics(topics=[row, robot_row])

    assert [topic.key for topic in msg.topics] == ["sound_events", "heard_sound"]
    assert msg.topics[0].recorded is True
    assert msg.topics[1].robot_scoped is True
    assert msg.topics[1].qos_transient_local is False
