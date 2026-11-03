#include "task_generator_gui/spawn_pedestrian_tool.hpp"

#include <chrono>
#include <cmath>
#include <memory>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/properties/string_property.hpp>

#include <pluginlib/class_list_macros.hpp>

namespace task_generator_gui
{
using namespace std::chrono_literals;

SpawnPedestrianTool::SpawnPedestrianTool()
{
  shortcut_key_ = 'p';

  target_node_property_ = new rviz_common::properties::StringProperty(
    "Target", "/arena_env_node",
    "Namespace of the task_generator node providing runtime/spawn_dynamic.",
    getPropertyContainer(), SLOT(updateClient()), this);

  model_property_ = new rviz_common::properties::StringProperty(
    "Model", "arenian",
    "Dynamic obstacle model name passed to runtime/spawn_dynamic.",
    getPropertyContainer());
}

SpawnPedestrianTool::~SpawnPedestrianTool() = default;

void SpawnPedestrianTool::onInitialize()
{
  PoseTool::onInitialize();
  setName("Spawn Pedestrian");

  service_node_ = std::make_shared<rclcpp::Node>("spawn_pedestrian_tool_node");
  service_node_->get_logger().set_level(rclcpp::Logger::Level::Warn);
  updateClient();
}

void SpawnPedestrianTool::updateClient()
{
  if (!service_node_) {
    return;
  }
  client_ = service_node_->create_client<arena_runtime_msgs::srv::SpawnDynamic>(
    target_node_property_->getStdString() + "/runtime/spawn_dynamic");
}

void SpawnPedestrianTool::onPoseSet(double x, double y, double theta)
{
  if (!client_) {
    updateClient();
  }

  auto request = std::make_shared<arena_runtime_msgs::srv::SpawnDynamic::Request>();
  request->model = model_property_->getStdString();
  request->use_pose = true;

  geometry_msgs::msg::PoseStamped & pose = request->pose;
  pose.header.frame_id = context_->getFixedFrame().toStdString();
  pose.header.stamp = service_node_->now();
  pose.pose.position.x = x;
  pose.pose.position.y = y;
  pose.pose.position.z = 0.0;
  pose.pose.orientation.x = 0.0;
  pose.pose.orientation.y = 0.0;
  pose.pose.orientation.z = std::sin(theta * 0.5);
  pose.pose.orientation.w = std::cos(theta * 0.5);

  if (!client_->wait_for_service(1s)) {
    RCLCPP_WARN(
      service_node_->get_logger(),
      "spawn_dynamic service not available at %s",
      client_->get_service_name());
    return;
  }

  auto future = client_->async_send_request(request);
  if (rclcpp::spin_until_future_complete(service_node_, future, 5s) !=
      rclcpp::FutureReturnCode::SUCCESS)
  {
    RCLCPP_ERROR(
      service_node_->get_logger(),
      "spawn_dynamic call to %s timed out",
      client_->get_service_name());
    return;
  }

  auto response = future.get();
  if (!response->success) {
    RCLCPP_WARN(
      service_node_->get_logger(),
      "spawn_dynamic rejected: %s", response->error_msg.c_str());
  } else {
    RCLCPP_INFO(
      service_node_->get_logger(),
      "spawned dynamic obstacle %s", response->id.c_str());
  }
}
}  // namespace task_generator_gui

PLUGINLIB_EXPORT_CLASS(task_generator_gui::SpawnPedestrianTool, rviz_common::Tool)
