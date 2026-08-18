cmake_minimum_required(VERSION 3.5)
project(arena_people_msgs)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(srv_files
  "srv/SpawnPedestrians.srv"
  "srv/DeletePedestrians.srv"
  "srv/UpdatePedestrians.srv"
  "srv/MovePedestrians.srv"
)

set(msg_files
  "msg/Gesture.msg"
  "msg/Pedestrian.msg"
  "msg/Pedestrians.msg"
  "msg/SpawnPedestrian.msg"
)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES builtin_interfaces geometry_msgs sensor_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()