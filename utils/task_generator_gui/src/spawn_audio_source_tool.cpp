#include "task_generator_gui/spawn_audio_source_tool.hpp"

#include <chrono>
#include <cmath>
#include <memory>

#include <rviz_common/display_context.hpp>
#include <rviz_common/properties/enum_property.hpp>
#include <rviz_common/properties/float_property.hpp>
#include <rviz_common/properties/string_property.hpp>

#include <pluginlib/class_list_macros.hpp>

namespace task_generator_gui
{
using namespace std::chrono_literals;

SpawnAudioSourceTool::SpawnAudioSourceTool()
{
  target_node_property_ = new rviz_common::properties::StringProperty(
    "Target", "/task_generator_node",
    "Node providing runtime/spawn_audio_source.",
    getPropertyContainer(), SLOT(updateClient()), this);

  mode_property_ = new rviz_common::properties::EnumProperty(
    "Mode", "Music", "Sound played by the spawned source.",
    getPropertyContainer());
  mode_property_->addOption("Music", 0);
  mode_property_->addOption("Alarm", 1);

  height_property_ = new rviz_common::properties::FloatProperty(
    "Height", 1.2, "Source Z coordinate in the RViz Fixed Frame, in metres.",
    getPropertyContainer());
  height_property_->setMin(0.0);
}

SpawnAudioSourceTool::~SpawnAudioSourceTool() = default;

void SpawnAudioSourceTool::onInitialize()
{
  PoseTool::onInitialize();
  setName("Spawn Radio");

  service_node_ = std::make_shared<rclcpp::Node>("spawn_audio_source_tool_node");
  service_node_->get_logger().set_level(rclcpp::Logger::Level::Info);
  updateClient();
}

void SpawnAudioSourceTool::updateClient()
{
  if (!service_node_) {
    return;
  }
  client_ = service_node_->create_client<task_generator_msgs::srv::SpawnAudioSource>(
    target_node_property_->getStdString() + "/runtime/spawn_audio_source");
}

void SpawnAudioSourceTool::onPoseSet(double x, double y, double theta)
{
  if (!client_) {
    updateClient();
  }

  auto request = std::make_shared<task_generator_msgs::srv::SpawnAudioSource::Request>();
  request->pose.header.frame_id = context_->getFixedFrame().toStdString();
  request->pose.header.stamp = service_node_->now();
  request->pose.pose.position.x = x;
  request->pose.pose.position.y = y;
  request->pose.pose.position.z = height_property_->getFloat();
  request->pose.pose.orientation.z = std::sin(theta / 2.0);
  request->pose.pose.orientation.w = std::cos(theta / 2.0);
  request->mode = mode_property_->getOptionInt() == 0 ? "music" : "alarm";

  if (!client_->wait_for_service(1s)) {
    RCLCPP_WARN(
      service_node_->get_logger(),
      "spawn_audio_source service not available at %s",
      client_->get_service_name());
    return;
  }

  auto future = client_->async_send_request(request);
  if (rclcpp::spin_until_future_complete(service_node_, future, 5s) !=
    rclcpp::FutureReturnCode::SUCCESS)
  {
    RCLCPP_ERROR(
      service_node_->get_logger(),
      "spawn_audio_source call to %s timed out",
      client_->get_service_name());
    return;
  }

  auto response = future.get();
  if (!response->success) {
    RCLCPP_WARN(
      service_node_->get_logger(),
      "spawn_audio_source rejected: %s", response->error_msg.c_str());
    return;
  }
  RCLCPP_INFO(
    service_node_->get_logger(),
    "spawned %s source %s",
    request->mode.c_str(), response->system_id.c_str());
}
}  // namespace task_generator_gui

PLUGINLIB_EXPORT_CLASS(task_generator_gui::SpawnAudioSourceTool, rviz_common::Tool)
