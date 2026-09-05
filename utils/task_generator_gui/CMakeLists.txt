cmake_minimum_required(VERSION 3.8)
project(task_generator_gui)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rviz_common REQUIRED)
find_package(rviz_default_plugins REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(task_generator_msgs REQUIRED)
find_package(world_generator_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)

set(CMAKE_AUTOMOC ON)
qt5_wrap_cpp(MOC_FILES
  include/task_generator_gui/task_generator_panel.hpp
  include/task_generator_gui/auditory_panel.hpp
  include/task_generator_gui/spawn_sound_tool.hpp
  include/task_generator_gui/spawn_microphone_tool.hpp
  include/task_generator_gui/spawn_pedestrian_tool.hpp
  include/task_generator_gui/world_generator_panel.hpp
  include/task_generator_gui/fleet_panel.hpp
  include/task_generator_gui/utils/sketch_edit.hpp
  include/task_generator_gui/utils/sketch_canvas.hpp
  include/Qt-MultiSelectComboBox/MultiSelectComboBox.h
  )

  add_library(task_generator_panel
  src/task_generator_panel.cpp
  src/task_generator_panel_getset_params.cpp
  src/fleet_panel.cpp
  src/auditory_panel.cpp
  src/task_generator_panel_dynamic_params.cpp
  src/utils/dynamic_param_tree.cpp
  src/utils/sketch_edit.cpp
  src/utils/sketch_canvas.cpp
  src/spawn_sound_tool.cpp
  src/spawn_microphone_tool.cpp
  src/spawn_pedestrian_tool.cpp
  src/world_generator_panel.cpp
  src/Qt-MultiSelectComboBox/MultiSelectComboBox.cpp
  ${MOC_FILES})
target_include_directories(task_generator_panel PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(task_generator_panel
  pluginlib
  rviz_common
  rviz_default_plugins
  geometry_msgs
  task_generator_msgs
  world_generator_msgs
  std_srvs
  rcl_interfaces
)
target_link_libraries(task_generator_panel Qt5::Widgets)

install(TARGETS task_generator_panel
        EXPORT export_task_generator_gui
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)
install(DIRECTORY include/
        DESTINATION include
)
install(FILES rviz_common_plugins.xml
        DESTINATION share/${PROJECT_NAME}
)
ament_export_include_directories(include)
ament_export_targets(export_task_generator_gui)
pluginlib_export_plugin_description_file(rviz_common rviz_common_plugins.xml)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
