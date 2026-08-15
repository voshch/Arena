"""Field-level schema validation for new task_generator_msgs types.

Instantiates each type, populates fields, reads them back, and asserts
equality. No ROS runtime required, this validates that rosidl-generated
types expose the correct field names and accept the right Python values.
Run after `colcon build` installs the generated Python bindings.
"""

def test_episode_record_fields():
    from rcl_interfaces.msg import Parameter, ParameterValue
    from task_generator_msgs.msg import EpisodeRecord

    r = EpisodeRecord()
    r.episode_id = 7
    r.world = "map1"
    r.seed = 12345
    r.tm_robots = "random"
    r.tm_obstacles = "parametrized"
    r.tm_modules = ["benchmark"]
    r.robots = ["husky", "jackal"]
    r.outcome_state = EpisodeRecord.SUCCESS
    r.outcome_info = "reached goal"
    r.goal_uuid = "abc-123"
    r.integrity = True

    p = Parameter()
    p.name = "static.n"
    p.value = ParameterValue()
    p.value.type = ParameterValue.PARAMETER_INTEGER
    p.value.integer_value = 4
    r.obstacles_params = [p]
    r.robots_params = []

    assert r.episode_id == 7
    assert r.world == "map1"
    assert r.seed == 12345
    assert r.tm_robots == "random"
    assert r.tm_obstacles == "parametrized"
    assert r.tm_modules == ["benchmark"]
    assert list(r.robots) == ["husky", "jackal"]
    assert r.outcome_state == EpisodeRecord.SUCCESS
    assert r.outcome_info == "reached goal"
    assert r.goal_uuid == "abc-123"
    assert r.integrity is True
    assert len(r.obstacles_params) == 1
    assert r.obstacles_params[0].name == "static.n"
    assert r.obstacles_params[0].value.integer_value == 4
    assert list(r.robots_params) == []

    assert EpisodeRecord.QUEUED == 0
    assert EpisodeRecord.RUNNING == 1
    assert EpisodeRecord.SUCCESS == 2
    assert EpisodeRecord.FAILED == 3
    assert EpisodeRecord.SKIPPED == 4
    assert EpisodeRecord.FATAL == 5


def test_reset_episode_srv():
    from task_generator_msgs.srv import ResetEpisode

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
    from task_generator_msgs.srv import QueryWorlds

    req = QueryWorlds.Request()
    res = QueryWorlds.Response()
    res.ids = ["map1", "map2"]

    assert res.ids == ["map1", "map2"]
    assert req is not None


def test_query_scenarios_srv():
    from task_generator_msgs.srv import QueryScenarios

    req = QueryScenarios.Request()
    req.world = "maze"
    res = QueryScenarios.Response()
    res.ids = ["scenario_a"]

    assert req.world == "maze"
    assert res.ids == ["scenario_a"]


def test_query_robots_srv():
    from task_generator_msgs.srv import QueryRobots

    res = QueryRobots.Response()
    res.ids = ["turtlebot3"]
    assert res.ids == ["turtlebot3"]


def test_query_static_obstacles_srv():
    from task_generator_msgs.srv import QueryStaticObstacles

    res = QueryStaticObstacles.Response()
    res.ids = ["box", "barrel"]
    assert res.ids == ["box", "barrel"]


def test_query_dynamic_obstacles_srv():
    from task_generator_msgs.srv import QueryDynamicObstacles

    res = QueryDynamicObstacles.Response()
    res.ids = ["pedestrian_adult"]
    assert res.ids == ["pedestrian_adult"]


def test_query_environments_srv():
    from task_generator_msgs.srv import QueryEnvironments

    res = QueryEnvironments.Response()
    res.ids = ["env_a"]
    assert res.ids == ["env_a"]


def test_query_parametrizeds_srv():
    from task_generator_msgs.srv import QueryParametrizeds

    res = QueryParametrizeds.Response()
    res.ids = ["param_set_1"]
    assert res.ids == ["param_set_1"]


def test_queue_episode_srv():
    from rcl_interfaces.msg import Parameter, ParameterValue
    from task_generator_msgs.srv import QueueEpisode

    req = QueueEpisode.Request()
    req.action = QueueEpisode.Request.MERGE
    req.tm_robots = "random"
    req.tm_obstacles = "parametrized"
    req.tm_modules = ["benchmark", "staged"]
    req.keep_modules = False
    req.world = "maze"

    p = Parameter()
    p.name = "static.n"
    p.value = ParameterValue()
    p.value.type = ParameterValue.PARAMETER_INTEGER
    p.value.integer_value = 5
    req.obstacles_params = [p]
    req.robots_params = []

    assert req.action == QueueEpisode.Request.MERGE
    assert QueueEpisode.Request.MERGE == 0
    assert req.tm_robots == "random"
    assert req.tm_obstacles == "parametrized"
    assert req.tm_modules == ["benchmark", "staged"]
    assert req.keep_modules is False
    assert req.world == "maze"
    assert len(req.obstacles_params) == 1
    assert req.obstacles_params[0].name == "static.n"

    res = QueueEpisode.Response()
    res.success = True
    res.error_msg = ""
    assert res.success is True


def test_get_task_modes_srv():
    from task_generator_msgs.srv import GetTaskModes

    res = GetTaskModes.Response()
    res.tm_robots = "random"
    res.tm_obstacles = "scenario"
    res.tm_modules = ["benchmark"]

    assert res.tm_robots == "random"
    assert res.tm_obstacles == "scenario"
    assert res.tm_modules == ["benchmark"]


def test_audio_system_state_msg():
    from task_generator_msgs.msg import AudioSystemState

    msg = AudioSystemState()
    msg.system_id = "building_alarm"
    msg.sound_type = "alarm"
    msg.asset_id = "alarm_loop"
    msg.loop = True
    msg.active = True
    msg.program_start_time.sec = 12
    msg.emitter_ids = ["environment:building_alarm:east"]

    assert msg.system_id == "building_alarm"
    assert msg.active is True
    assert msg.program_start_time.sec == 12
    assert msg.emitter_ids == ["environment:building_alarm:east"]


def test_set_audio_system_srv():
    from task_generator_msgs.srv import SetAudioSystem

    req = SetAudioSystem.Request()
    req.system_id = "building_alarm"
    req.active = True
    assert req.system_id == "building_alarm"
    assert req.active is True

    res = SetAudioSystem.Response()
    res.success = True
    res.error_msg = ""
    assert res.success is True


def test_spawn_audio_source_srv():
    from task_generator_msgs.srv import SpawnAudioSource

    req = SpawnAudioSource.Request()
    req.pose.header.frame_id = "map"
    req.mode = "music"
    req.customize_playback = True
    req.asset_id = "custom_radio"
    req.source_volume_db = 70.0
    req.loop = True
    req.initially_active = False
    assert req.pose.header.frame_id == "map"
    assert req.mode == "music"
    assert req.asset_id == "custom_radio"
    assert req.initially_active is False

    res = SpawnAudioSource.Response()
    res.system_id = "runtime_music_1"
    res.success = True
    assert res.system_id == "runtime_music_1"
    assert res.success is True


def test_spawn_static_srv():
    from geometry_msgs.msg import PoseStamped
    from task_generator_msgs.srv import SpawnStatic

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
    from task_generator_msgs.srv import SpawnDynamic

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


def test_spawn_microphone_srv():
    from geometry_msgs.msg import PointStamped
    from task_generator_msgs.srv import SpawnMicrophone

    req = SpawnMicrophone.Request()
    req.position = PointStamped()
    req.position.header.frame_id = "map"
    req.position.point.x = 2.0
    req.position.point.y = 3.0
    req.position.point.z = 1.5
    req.placement = "placed"
    req.attached_frame = "robot/base_link"

    assert req.position.header.frame_id == "map"
    assert req.position.point.z == 1.5
    assert req.placement == "placed"
    assert req.attached_frame == "robot/base_link"

    res = SpawnMicrophone.Response()
    res.listener_id = "microphone1"
    res.zone = "reception"
    res.attached_frame = "robot/base_link"
    res.success = True
    res.error_msg = ""

    assert res.listener_id == "microphone1"
    assert res.zone == "reception"
    assert res.attached_frame == "robot/base_link"
    assert res.success is True


def test_remove_audio_system_srv():
    from task_generator_msgs.srv import RemoveAudioSystem

    req = RemoveAudioSystem.Request()
    req.system_id = "runtime_music_1"
    res = RemoveAudioSystem.Response()
    res.success = True

    assert req.system_id == "runtime_music_1"
    assert res.success is True


def test_remove_microphone_srv():
    from task_generator_msgs.srv import RemoveMicrophone

    req = RemoveMicrophone.Request()
    req.listener_id = "microphone:map:placed:1"
    res = RemoveMicrophone.Response()
    res.success = True

    assert req.listener_id == "microphone:map:placed:1"
    assert res.success is True


def test_spawn_robot_srv():
    from diagnostic_msgs.msg import KeyValue
    from geometry_msgs.msg import PoseStamped
    from task_generator_msgs.srv import SpawnRobot

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
    from task_generator_msgs.srv import DespawnRobot

    req = DespawnRobot.Request()
    req.name = "turtlebot3_0"

    assert req.name == "turtlebot3_0"

    res = DespawnRobot.Response()
    res.success = True
    res.error_msg = ""

    assert res.success is True


def test_robot_queue_msg():
    from task_generator_msgs.msg import RobotDescriptor, RobotQueue

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
    from task_generator_msgs.msg import RobotCap, RobotDescriptor, RobotFleet, RobotState

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
    from task_generator_msgs.action import RunEpisode

    goal = RunEpisode.Goal()
    goal.world = "maze"
    assert goal.world == "maze"


def test_run_episode_action_result():
    from task_generator_msgs.action import RunEpisode

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
    from task_generator_msgs.action import RunEpisode

    fb = RunEpisode.Feedback()
    fb.state = RunEpisode.Feedback.STARTED
    assert fb.state == RunEpisode.Feedback.STARTED
    assert RunEpisode.Feedback.STARTED == 1
