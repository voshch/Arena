#include "task_generator_gui/spawn_microphone_tool.hpp"

#include <chrono>
#include <memory>
#include <string>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/properties/float_property.hpp>
#include <rviz_common/properties/string_property.hpp>

#include <pluginlib/class_list_macros.hpp>

namespace task_generator_gui
{
using namespace std::chrono_literals;

SpawnMicrophoneTool::SpawnMicrophoneTool()
{
  shortcut_key_ = 'm';

  target_node_property_ = new rviz_common::properties::StringProperty(
    "Target", "/task_generator_node",
    "Namespace providing the acoustic runtime/spawn_microphone service.",
    getPropertyContainer(), SLOT(updateClient()), this);

  height_property_ = new rviz_common::properties::FloatProperty(
    "Height", 1.5,
    "Microphone Z coordinate in the RViz Fixed Frame, in metres.",
    getPropertyContainer());
  height_property_->setMin(0.0);

  attached_frame_property_ = new rviz_common::properties::StringProperty(
    "Attach TF Frame", "",
    "Optional TF frame. Empty creates a fixed microphone. A frame makes the microphone follow it.",
    getPropertyContainer());
}

SpawnMicrophoneTool::~SpawnMicrophoneTool() = default;

void SpawnMicrophoneTool::onInitialize()
{
  PoseTool::onInitialize();
  setName("Spawn Microphone");

  service_node_ = std::make_shared<rclcpp::Node>("spawn_microphone_tool_node");
  service_node_->get_logger().set_level(rclcpp::Logger::Level::Info);
  updateClient();
}

void SpawnMicrophoneTool::updateClient()
{
  if (!service_node_) {
    return;
  }
  client_ = service_node_->create_client<task_generator_msgs::srv::SpawnMicrophone>(
    target_node_property_->getStdString() + "/runtime/spawn_microphone");
}

void SpawnMicrophoneTool::onPoseSet(double x, double y, double theta)
{
  static_cast<void>(theta);

  if (!client_) {
    updateClient();
  }

  auto request = std::make_shared<task_generator_msgs::srv::SpawnMicrophone::Request>();
  request->position.header.frame_id = context_->getFixedFrame().toStdString();
  request->position.header.stamp = service_node_->now();
  request->position.point.x = x;
  request->position.point.y = y;
  request->position.point.z = height_property_->getFloat();
  request->placement = "placed";
  request->attached_frame = attached_frame_property_->getStdString();

  if (!client_->wait_for_service(1s)) {
    RCLCPP_WARN(
      service_node_->get_logger(),
      "spawn_microphone service not available at %s. Is auditory simulation enabled?",
      client_->get_service_name());
    return;
  }

  auto future = client_->async_send_request(request);
  if (rclcpp::spin_until_future_complete(service_node_, future, 5s) !=
      rclcpp::FutureReturnCode::SUCCESS)
  {
    RCLCPP_ERROR(
      service_node_->get_logger(),
      "spawn_microphone call to %s timed out",
      client_->get_service_name());
    return;
  }

  auto response = future.get();
  if (!response->success) {
    RCLCPP_WARN(
      service_node_->get_logger(),
      "spawn_microphone rejected: %s", response->error_msg.c_str());
  } else {
    RCLCPP_INFO(
      service_node_->get_logger(),
      "spawned microphone %s%s%s",
      response->listener_id.c_str(),
      response->attached_frame.empty() ? "" : " attached to ",
      response->attached_frame.c_str());
  }
}
}  // namespace task_generator_gui

PLUGINLIB_EXPORT_CLASS(task_generator_gui::SpawnMicrophoneTool, rviz_common::Tool)
