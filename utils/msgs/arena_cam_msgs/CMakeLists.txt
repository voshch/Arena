cmake_minimum_required(VERSION 3.5)
project(arena_cam_msgs)

find_package(ament_cmake_auto REQUIRED)
find_package(rosidl_default_generators REQUIRED)

ament_auto_find_build_dependencies()

rosidl_generate_interfaces(${PROJECT_NAME}
  srv/Play.srv
)

ament_export_dependencies(rosidl_default_runtime)
ament_auto_package()
