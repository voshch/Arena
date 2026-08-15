#ifndef TASK_GENERATOR_GUI_SPAWN_MICROPHONE_TOOL_HPP
#define TASK_GENERATOR_GUI_SPAWN_MICROPHONE_TOOL_HPP

#include <memory>
#include <string>

#include <QObject>

#include <rclcpp/rclcpp.hpp>
#include <rviz_default_plugins/tools/pose/pose_tool.hpp>

#include "task_generator_msgs/srv/spawn_microphone.hpp"

namespace rviz_common
{
namespace properties
{
class FloatProperty;
class StringProperty;
}
}  // namespace rviz_common

namespace task_generator_gui
{
class SpawnMicrophoneTool : public rviz_default_plugins::tools::PoseTool
{
  Q_OBJECT

public:
  SpawnMicrophoneTool();
  ~SpawnMicrophoneTool() override;

  void onInitialize() override;

protected:
  void onPoseSet(double x, double y, double theta) override;

private Q_SLOTS:
  void updateClient();

private:
  rviz_common::properties::StringProperty * target_node_property_;
  rviz_common::properties::FloatProperty * height_property_;
  rviz_common::properties::StringProperty * attached_frame_property_;

  std::shared_ptr<rclcpp::Node> service_node_;
  rclcpp::Client<task_generator_msgs::srv::SpawnMicrophone>::SharedPtr client_;
};
}  // namespace task_generator_gui

#endif  // TASK_GENERATOR_GUI_SPAWN_MICROPHONE_TOOL_HPP
