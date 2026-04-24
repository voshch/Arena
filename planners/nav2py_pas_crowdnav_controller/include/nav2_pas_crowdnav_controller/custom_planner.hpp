#ifndef NAV2_PAS_CROWDNAV_CONTROLLER__CUSTOM_PLANNER_HPP_
#define NAV2_PAS_CROWDNAV_CONTROLLER__CUSTOM_PLANNER_HPP_

#include <string>
#include <vector>
#include <memory>

#include "nav2_core/controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "pluginlib/class_loader.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"

namespace nav2_pas_crowdnav_controller
{

class PasCrowdNavController : public nav2_core::Controller
{
public:
  PasCrowdNavController() = default;
  ~PasCrowdNavController() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;
  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;

  void setPlan(const nav_msgs::msg::Path & path) override;

protected:
  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::string plugin_name_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  rclcpp::Logger logger_ {rclcpp::get_logger("PasCrowdNavController")};

  // Controller Parameters
  double lookahead_dist_;
  double v_max_;
  double v_min_;           // NEW: Minimum speed
  double slowdown_dist_;   // NEW: Distance to start slowing down
  double desired_linear_vel_;
  double rotation_threshold_; 
  rclcpp::Duration transform_tolerance_ {0, 0};

  nav_msgs::msg::Path global_plan_;
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Path>> local_plan_pub_;
};

}  // namespace nav2_pas_crowdnav_controller

#endif
