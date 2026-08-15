cmake_minimum_required(VERSION 3.8)
project(arena_robots_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/GotoPose.action"
  "action/PlayGesture.action"
  "action/ReachPose.action"
  "msg/Acoustics.msg"
  "msg/Power.msg"
  "msg/Energy.msg"
  "msg/CollisionEvent.msg"
  "msg/CollisionEvents.msg"
  DEPENDENCIES geometry_msgs action_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
