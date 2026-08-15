#include "task_generator_gui/spawn_microphone_tool.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/parameter_client.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/properties/float_property.hpp>
#include <rviz_common/properties/string_property.hpp>

#include <pluginlib/class_list_macros.hpp>

namespace task_generator_gui
{
using namespace std::chrono_literals;

SpawnMicrophoneTool::SpawnMicrophoneTool()
{
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
    selectPlaybackListener(response->listener_id);
    RCLCPP_INFO(
      service_node_->get_logger(),
      "spawned and selected microphone %s in zone %s%s%s",
      response->listener_id.c_str(), response->zone.c_str(),
      response->attached_frame.empty() ? "" : " attached to ",
      response->attached_frame.c_str());
  }
}

void SpawnMicrophoneTool::selectPlaybackListener(const std::string & listener_id)
{
  const std::vector<std::string> playback_nodes = {
    "human_sound_playback",
    "robot_sound_node",
    "environmental_sound_playback",
  };
  for (const auto & node_name : playback_nodes) {
    const auto remote_node =
      target_node_property_->getStdString() + "/" + node_name;
    auto parameters = std::make_shared<rclcpp::AsyncParametersClient>(
      service_node_, remote_node);
    if (!parameters->wait_for_service(500ms)) {
      RCLCPP_WARN(
        service_node_->get_logger(),
        "parameter service not available for %s", remote_node.c_str());
      continue;
    }
    auto future = parameters->set_parameters({
      rclcpp::Parameter("listener_mode", "selected"),
      rclcpp::Parameter("listener_id", listener_id),
    });
    if (rclcpp::spin_until_future_complete(service_node_, future, 2s) !=
      rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_WARN(
        service_node_->get_logger(),
        "listener selection timed out for %s", remote_node.c_str());
      continue;
    }
    for (const auto & result : future.get()) {
      if (!result.successful) {
        RCLCPP_WARN(
          service_node_->get_logger(),
          "listener selection failed for %s: %s",
          remote_node.c_str(), result.reason.c_str());
      }
    }
  }
}
}  // namespace task_generator_gui

PLUGINLIB_EXPORT_CLASS(task_generator_gui::SpawnMicrophoneTool, rviz_common::Tool)
