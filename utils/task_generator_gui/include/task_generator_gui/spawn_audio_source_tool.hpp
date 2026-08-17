#ifndef TASK_GENERATOR_GUI_SPAWN_AUDIO_SOURCE_TOOL_HPP
#define TASK_GENERATOR_GUI_SPAWN_AUDIO_SOURCE_TOOL_HPP

#include <memory>

#include <QObject>

#include <rclcpp/rclcpp.hpp>
#include <rviz_default_plugins/tools/pose/pose_tool.hpp>

#include "task_generator_msgs/srv/spawn_audio_source.hpp"

namespace rviz_common
{
namespace properties
{
class EnumProperty;
class BoolProperty;
class FloatProperty;
class StringProperty;
}
}  // namespace rviz_common

namespace task_generator_gui
{
class SpawnAudioSourceTool : public rviz_default_plugins::tools::PoseTool
{
  Q_OBJECT

public:
  SpawnAudioSourceTool();
  ~SpawnAudioSourceTool() override;

  void onInitialize() override;

protected:
  void onPoseSet(double x, double y, double theta) override;

private Q_SLOTS:
  void updateClient();
  void updateModeDefaults();

private:
  rviz_common::properties::StringProperty * target_node_property_;
  rviz_common::properties::EnumProperty * mode_property_;
  rviz_common::properties::FloatProperty * height_property_;
  rviz_common::properties::BoolProperty * customize_property_;
  rviz_common::properties::StringProperty * asset_property_;
  rviz_common::properties::FloatProperty * volume_property_;
  rviz_common::properties::BoolProperty * loop_property_;
  rviz_common::properties::BoolProperty * initially_active_property_;

  std::shared_ptr<rclcpp::Node> service_node_;
  rclcpp::Client<task_generator_msgs::srv::SpawnAudioSource>::SharedPtr client_;
};
}  // namespace task_generator_gui

#endif  // TASK_GENERATOR_GUI_SPAWN_AUDIO_SOURCE_TOOL_HPP
