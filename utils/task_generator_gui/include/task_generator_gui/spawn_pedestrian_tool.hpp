#ifndef TASK_GENERATOR_GUI_SPAWN_PEDESTRIAN_TOOL_HPP
#define TASK_GENERATOR_GUI_SPAWN_PEDESTRIAN_TOOL_HPP

#include <memory>
#include <string>

#include <QObject>

#include <rclcpp/rclcpp.hpp>
#include <rviz_default_plugins/tools/pose/pose_tool.hpp>

#include "arena_runtime_msgs/srv/spawn_dynamic.hpp"

namespace rviz_common
{
namespace properties
{
class StringProperty;
}
}  // namespace rviz_common

namespace task_generator_gui
{
class SpawnPedestrianTool : public rviz_default_plugins::tools::PoseTool
{
  Q_OBJECT

public:
  SpawnPedestrianTool();
  ~SpawnPedestrianTool() override;

  void onInitialize() override;

protected:
  void onPoseSet(double x, double y, double theta) override;

private Q_SLOTS:
  void updateClient();

private:
  rviz_common::properties::StringProperty * target_node_property_;
  rviz_common::properties::StringProperty * model_property_;

  std::shared_ptr<rclcpp::Node> service_node_;
  rclcpp::Client<arena_runtime_msgs::srv::SpawnDynamic>::SharedPtr client_;
};
}  // namespace task_generator_gui

#endif  // TASK_GENERATOR_GUI_SPAWN_PEDESTRIAN_TOOL_HPP
