cmake_minimum_required(VERSION 3.5)
project(task_generator_msgs)

find_package(ament_cmake_auto REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rcl_interfaces REQUIRED)

ament_auto_find_build_dependencies()

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/AdapterDisplay.msg
  msg/AdapterEntry.msg
  msg/AdapterVizManifest.msg
  msg/EpisodeRecord.msg
  msg/RobotCap.msg
  msg/RobotDescriptor.msg
  msg/RobotFleet.msg
  msg/RobotQueue.msg
  msg/RobotState.msg
  msg/SemanticEntityState.msg
  msg/SemanticSnapshot.msg
  srv/DespawnRobot.srv
  srv/GetTaskModes.srv
  srv/Pause.srv
  srv/QueryDynamicObstacles.srv
  srv/QueryEnvironments.srv
  srv/QueryParametrizeds.srv
  srv/QueryRobots.srv
  srv/QueryScenarios.srv
  srv/QueryStaticObstacles.srv
  srv/QueryTaskModes.srv
  srv/QueryWorlds.srv
  srv/QueueEpisode.srv
  srv/ResetEpisode.srv
  srv/SetSemantic.srv
  srv/SpawnDynamic.srv
  srv/SpawnRobot.srv
  srv/SpawnStatic.srv
  action/RunEpisode.action
  DEPENDENCIES std_msgs geometry_msgs diagnostic_msgs builtin_interfaces rcl_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_auto_package()
