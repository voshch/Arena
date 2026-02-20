#include "nav2_pas_crowdnav_controller/custom_planner.hpp"
#include "nav2_core/exceptions.hpp"
#include "nav2_util/node_utils.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include <cmath>
#include <algorithm>

using nav2_util::declare_parameter_if_not_declared;

namespace nav2_pas_crowdnav_controller
{

void PasCrowdNavController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent;
  auto node = node_.lock();
  costmap_ros_ = costmap_ros;
  tf_ = tf;
  plugin_name_ = name;
  logger_ = node->get_logger();

  declare_parameter_if_not_declared(node, plugin_name_ + ".lookahead_dist", rclcpp::ParameterValue(0.6));
  declare_parameter_if_not_declared(node, plugin_name_ + ".v_max", rclcpp::ParameterValue(0.4));
  declare_parameter_if_not_declared(node, plugin_name_ + ".v_min", rclcpp::ParameterValue(0.1));
  declare_parameter_if_not_declared(node, plugin_name_ + ".slowdown_dist", rclcpp::ParameterValue(0.8));
  declare_parameter_if_not_declared(node, plugin_name_ + ".rotation_threshold", rclcpp::ParameterValue(0.35));
  declare_parameter_if_not_declared(node, plugin_name_ + ".transform_tolerance", rclcpp::ParameterValue(0.1));

  node->get_parameter(plugin_name_ + ".lookahead_dist", lookahead_dist_);
  node->get_parameter(plugin_name_ + ".v_max", v_max_);
  node->get_parameter(plugin_name_ + ".v_min", v_min_);
  node->get_parameter(plugin_name_ + ".slowdown_dist", slowdown_dist_);
  node->get_parameter(plugin_name_ + ".rotation_threshold", rotation_threshold_);
  desired_linear_vel_ = v_max_;
  
  double transform_tolerance;
  node->get_parameter(plugin_name_ + ".transform_tolerance", transform_tolerance);
  transform_tolerance_ = rclcpp::Duration::from_seconds(transform_tolerance);

  local_plan_pub_ = node->create_publisher<nav_msgs::msg::Path>("local_plan", 1);

  RCLCPP_INFO(logger_, "PasCrowdNavController configured with Velocity Ramping.");
}

void PasCrowdNavController::cleanup() { local_plan_pub_.reset(); }
void PasCrowdNavController::activate() { local_plan_pub_->on_activate(); }
void PasCrowdNavController::deactivate() { local_plan_pub_->on_deactivate(); }

void PasCrowdNavController::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  if (speed_limit <= 0.0) { 
    desired_linear_vel_ = v_max_;
  } else {
    desired_linear_vel_ = percentage ? (v_max_ * speed_limit / 100.0) : speed_limit;
  }
}

void PasCrowdNavController::setPlan(const nav_msgs::msg::Path & path)
{
  global_plan_ = path;
}

geometry_msgs::msg::TwistStamped PasCrowdNavController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & velocity,
  nav2_core::GoalChecker * goal_checker)
{
  geometry_msgs::msg::TwistStamped cmd_vel;
  cmd_vel.header.frame_id = pose.header.frame_id;
  cmd_vel.header.stamp = costmap_ros_->get_clock()->now(); 
  cmd_vel.twist.linear.x = 0.0;
  cmd_vel.twist.angular.z = 0.0;

  if (global_plan_.poses.empty()) return cmd_vel;

  if (goal_checker->isGoalReached(pose.pose, global_plan_.poses.back().pose, velocity)) {
    return cmd_vel; 
  }

  // 1. Path Transformation
  nav_msgs::msg::Path local_path;
  local_path.header.frame_id = costmap_ros_->getBaseFrameID();
  local_path.header.stamp = cmd_vel.header.stamp;
  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_->lookupTransform(local_path.header.frame_id, global_plan_.header.frame_id, tf2::TimePointZero);
    for (const auto & global_pose : global_plan_.poses) {
      geometry_msgs::msg::PoseStamped local_pose;
      tf2::doTransform(global_pose, local_pose, transform);
      local_path.poses.push_back(local_pose);
    }
  } catch (tf2::TransformException & ex) { return cmd_vel; }
  local_plan_pub_->publish(local_path);

  // 2. Select Lookahead Point
  geometry_msgs::msg::PoseStamped target_pose = local_path.poses.back(); 
  for (const auto & pt : local_path.poses) {
    if (std::hypot(pt.pose.position.x, pt.pose.position.y) >= lookahead_dist_) {
      target_pose = pt;
      break;
    }
  }

  // 3. Velocity Logic with Distance Ramping
  double angle_to_target = std::atan2(target_pose.pose.position.y, target_pose.pose.position.x);
  double dist_to_goal = std::hypot(
    global_plan_.poses.back().pose.position.x - pose.pose.position.x,
    global_plan_.poses.back().pose.position.y - pose.pose.position.y);

  if (std::abs(angle_to_target) > rotation_threshold_ && dist_to_goal > 0.4) {
    // Sharp turn mode
    cmd_vel.twist.linear.x = 0.0;
    cmd_vel.twist.angular.z = std::clamp(2.5 * angle_to_target, -1.2, 1.2); 
  } else {
    // Smooth navigation mode
    double L_sq = std::pow(target_pose.pose.position.x, 2) + std::pow(target_pose.pose.position.y, 2);
    double curvature = (2.0 * target_pose.pose.position.y) / std::max(L_sq, 0.01);
    
    // RAMP LOGIC: If close to goal, base speed ramps down from v_max to v_min
    double dist_ratio = std::clamp(dist_to_goal / slowdown_dist_, 0.0, 1.0);
    double ramped_v = v_min_ + (desired_linear_vel_ - v_min_) * dist_ratio;
    
    // TURN SCALER: Still slow down for minor corrections
    double turn_scaler = std::max(0.4, 1.0 - (std::abs(angle_to_target) / rotation_threshold_));
    
    cmd_vel.twist.linear.x = ramped_v * turn_scaler;
    cmd_vel.twist.angular.z = cmd_vel.twist.linear.x * curvature;
  }

  return cmd_vel;
}

}  // namespace nav2_pas_crowdnav_controller

PLUGINLIB_EXPORT_CLASS(nav2_pas_crowdnav_controller::PasCrowdNavController, nav2_core::Controller)
