#include "task_generator_gui/spawn_audio_source_tool.hpp"

#include <cmath>
#include <exception>
#include <memory>

#include <rviz_common/display_context.hpp>
#include <rviz_common/properties/bool_property.hpp>
#include <rviz_common/properties/enum_property.hpp>
#include <rviz_common/properties/float_property.hpp>
#include <rviz_common/properties/string_property.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>

#include <pluginlib/class_list_macros.hpp>

namespace task_generator_gui
{
SpawnAudioSourceTool::SpawnAudioSourceTool()
{
  shortcut_key_ = 'r';

  target_node_property_ = new rviz_common::properties::StringProperty(
    "Target", "/task_generator_node",
    "Node providing runtime/spawn_audio_source.",
    getPropertyContainer(), SLOT(updateClient()), this);

  mode_property_ = new rviz_common::properties::EnumProperty(
    "Mode", "Music", "Sound played by the spawned source.",
    getPropertyContainer());
  mode_property_->addOption("Music", 0);
  mode_property_->addOption("Alarm", 1);
  connect(
    mode_property_, SIGNAL(changed()), this, SLOT(updateModeDefaults()));

  height_property_ = new rviz_common::properties::FloatProperty(
    "Height", 1.2, "Source Z coordinate in the RViz Fixed Frame, in metres.",
    getPropertyContainer());
  height_property_->setMin(0.0);

  customize_property_ = new rviz_common::properties::BoolProperty(
    "Custom Playback", false,
    "Use the asset, volume, loop, and initial-state properties below.",
    getPropertyContainer());
  asset_property_ = new rviz_common::properties::StringProperty(
    "Asset ID", "",
    "Catalog asset ID. Empty uses radio_loop or alarm_loop.",
    getPropertyContainer());
  volume_property_ = new rviz_common::properties::FloatProperty(
    "Source Volume", 62.0,
    "Source level in dB.", getPropertyContainer());
  volume_property_->setMin(-120.0);
  volume_property_->setMax(160.0);
  loop_property_ = new rviz_common::properties::BoolProperty(
    "Loop", true, "Loop the selected WAV.", getPropertyContainer());
  initially_active_property_ = new rviz_common::properties::BoolProperty(
    "Start Immediately", true,
    "Start emission and playback when the source is spawned.",
    getPropertyContainer());
}

SpawnAudioSourceTool::~SpawnAudioSourceTool() = default;

void SpawnAudioSourceTool::onInitialize()
{
  PoseTool::onInitialize();
  setName("Spawn Radio");

  auto node_abstraction = context_->getRosNodeAbstraction().lock();
  service_node_ = node_abstraction->get_raw_node();
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

void SpawnAudioSourceTool::updateModeDefaults()
{
  if (!volume_property_ || !customize_property_
    || customize_property_->getBool())
  {
    return;
  }
  volume_property_->setFloat(mode_property_->getOptionInt() == 0 ? 62.0 : 88.0);
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
  request->customize_playback = customize_property_->getBool();
  request->asset_id = asset_property_->getStdString();
  request->source_volume_db = volume_property_->getFloat();
  request->loop = loop_property_->getBool();
  request->initially_active = initially_active_property_->getBool();

  if (!client_->service_is_ready()) {
    RCLCPP_WARN(
      service_node_->get_logger(),
      "spawn_audio_source service not available at %s",
      client_->get_service_name());
    return;
  }

  const auto logger = service_node_->get_logger();
  client_->async_send_request(
    request,
    [logger, mode = request->mode](
      rclcpp::Client<task_generator_msgs::srv::SpawnAudioSource>::SharedFuture future)
    {
      try {
        const auto response = future.get();
        if (!response->success) {
          RCLCPP_WARN(
            logger, "spawn_audio_source rejected: %s",
            response->error_msg.c_str());
          return;
        }
        RCLCPP_INFO(
          logger, "spawned %s source %s",
          mode.c_str(), response->system_id.c_str());
      } catch (const std::exception & exception) {
        RCLCPP_ERROR(logger, "spawn_audio_source failed: %s", exception.what());
      }
    });
}
}  // namespace task_generator_gui

PLUGINLIB_EXPORT_CLASS(task_generator_gui::SpawnAudioSourceTool, rviz_common::Tool)
