"""Body sizes and clearances the generators place robots and pedestrians with."""

# mirrors the runtime `task.episode.spawn.obstacle_clearance`
RUNTIME_SAFE_DIST = 0.35
ROBOT_RADIUS = 0.35
PED_RADIUS = RUNTIME_SAFE_DIST
CLEARANCE = 0.1  # free space kept between a body and a wall
