cmake_minimum_required(VERSION 3.5)
project(arena_runtime_msgs)

find_package(ament_cmake_auto REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)

ament_auto_find_build_dependencies()

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/AdapterDisplay.msg
  msg/AdapterEntry.msg
  msg/AdapterPlugin.msg
  msg/AdapterVizManifest.msg
  msg/EnvRecord.msg
  msg/EnvRegistry.msg
  msg/EpisodeRecord.msg
  msg/Heartbeat.msg
  msg/HoldEntry.msg
  msg/HoldRegistry.msg
  msg/LockstepChannel.msg
  msg/LockstepChannelCount.msg
  msg/LockstepHeartbeat.msg
  msg/LockstepRegistration.msg
  msg/LockstepStatus.msg
  msg/RecordedTopic.msg
  msg/RecordedTopics.msg
  msg/RobotCap.msg
  msg/RobotDescriptor.msg
  msg/RobotFleet.msg
  msg/RobotQueue.msg
  msg/RobotState.msg
  msg/SemanticEntityState.msg
  msg/SemanticSnapshot.msg
  msg/ShutdownRequest.msg
  msg/SimState.msg
  msg/WorldExtent.msg
  srv/CleanupEnv.srv
  srv/ConfirmWorld.srv
  srv/DespawnEnv.srv
  srv/DespawnRobot.srv
  srv/GetTaskModes.srv
  srv/LifecycleHold.srv
  srv/LifecycleStep.srv
  srv/LifecycleUnpauseWindow.srv
  srv/LockstepRegister.srv
  srv/LockstepStart.srv
  srv/MoveEntity.srv
  srv/QueryDynamicObstacles.srv
  srv/QueryEnvironments.srv
  srv/QueryParametrizeds.srv
  srv/QueryRobots.srv
  srv/QueryScenarios.srv
  srv/QueryStaticObstacles.srv
  srv/QueryTaskModes.srv
  srv/QueryWorlds.srv
  srv/QueueEpisode.srv
  srv/RegisterEnv.srv
  srv/RemoveSound.srv
  srv/ResetEpisode.srv
  srv/SetSemantic.srv
  srv/SpawnDynamic.srv
  srv/SpawnEnv.srv
  srv/SpawnRobot.srv
  srv/SpawnSound.srv
  srv/SpawnStatic.srv
  srv/SubmitTask.srv
  action/RunEpisode.action
  action/RunTask.action
  DEPENDENCIES std_msgs geometry_msgs diagnostic_msgs builtin_interfaces rcl_interfaces sensor_msgs
)


ament_export_dependencies(rosidl_default_runtime)
ament_auto_package()