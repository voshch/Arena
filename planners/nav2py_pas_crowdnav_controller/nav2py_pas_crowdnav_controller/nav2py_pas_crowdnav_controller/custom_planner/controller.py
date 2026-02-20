#!/usr/bin/env python3

import numpy as np
import logging
import math
from scipy.spatial.transform import Rotation

class Controller:
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger
        self.global_path = None
        self.goal_pose = None
        
        # --- Pure Pursuit Parameters ---
        self.lookahead_dist = 1.0  # meters: How far ahead to look on the path
        self.v_max = 0.5           # m/s: Forward speed
        self.goal_tolerance = 0.2  # meters: When to stop

    def setPath(self, global_plan):
        """Called by C++ whenever Nav2 generates a new global path."""
        self.global_path = global_plan
        if len(global_plan.poses) > 0:
            self.goal_pose = global_plan.poses[-1]
            if self.logger:
                self.logger.info(f"New path received with {len(global_plan.poses)} waypoints.")

    def scan_callback(self, msg):
        """Called by C++ whenever a new LiDAR scan arrives."""
        # We will leave this empty for now! Pure Pursuit just follows the line.
        pass

    def setSpeedLimit(self, speed_limit, is_percentage):
        pass

    def computeVelocityCommands(self, pose):
        """Called by C++ at 20Hz to get the [vx, vy, omega] command."""
        
        # 1. Safety Check: Do we have a path?
        if self.global_path is None or len(self.global_path.poses) == 0:
            return 0.0, 0.0

        # 2. Extract Robot State
        rx = pose.position.x
        ry = pose.position.y
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        _, _, r_theta = Rotation.from_quat(quat).as_euler('xyz')

        # 3. Check if Goal Reached
        gx = self.goal_pose.position.x
        gy = self.goal_pose.position.y
        dist_to_goal = math.hypot(gx - rx, gy - ry)
        if dist_to_goal < self.goal_tolerance:
            if self.logger:
                self.logger.info("Goal reached! Stopping.")
            return 0.0, 0.0

        # 4. Find the Target Point on the Path
        target_pt = self._get_lookahead_point(rx, ry)

        # 5. Transform Target Point to Robot's Local Frame
        dx = target_pt[0] - rx
        dy = target_pt[1] - ry
        
        # Rotate by -r_theta to get local coordinates
        local_x = dx * math.cos(-r_theta) - dy * math.sin(-r_theta)
        local_y = dx * math.sin(-r_theta) + dy * math.cos(-r_theta)

        # 6. Pure Pursuit Math (Curvature Calculation)
        # The mathematical formula for Pure Pursuit curvature is: gamma = 2 * y / L^2
        distance_to_target = math.hypot(local_x, local_y)
        
        # Prevent division by zero if the point is exactly under the robot
        if distance_to_target < 0.01:
            return 0.0, 0.0
            
        curvature = (2.0 * local_y) / (distance_to_target ** 2)

        # 7. Generate Velocities
        # Slow down slightly as we get very close to the final goal
        current_speed = self.v_max if dist_to_goal > 1.0 else (self.v_max * dist_to_goal)
        current_speed = max(0.1, current_speed) # Don't go slower than 0.1 m/s

        linear_x = current_speed
        angular_z = current_speed * curvature

        # Clip angular velocity to realistic Jackal hardware limits
        angular_z = np.clip(angular_z, -1.5, 1.5)

        return float(linear_x), float(angular_z)

    def _get_lookahead_point(self, rx, ry):
        """Helper function to find the point on the path that is `lookahead_dist` away."""
        closest_dist = float('inf')
        closest_idx = 0
        
        # Find the absolute closest point on the path to the robot
        for i, waypoint in enumerate(self.global_path.poses):
            dist = math.hypot(waypoint.position.x - rx, waypoint.position.y - ry)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
                
        # Search forward from the closest point to find the lookahead point
        for i in range(closest_idx, len(self.global_path.poses)):
            dist = math.hypot(self.global_path.poses[i].position.x - rx, 
                              self.global_path.poses[i].position.y - ry)
            if dist >= self.lookahead_dist:
                return [self.global_path.poses[i].position.x, self.global_path.poses[i].position.y]
                
        # If we run out of path, just aim for the very last point (the goal)
        return [self.global_path.poses[-1].position.x, self.global_path.poses[-1].position.y]
