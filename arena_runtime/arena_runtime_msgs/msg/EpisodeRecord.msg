# Same id may appear repeatedly across publications as outcome resolves;
# subscribers dedup by episode_id and keep the latest version per id.
uint32 episode_id
string world
int64 seed
# The task runtime that ran the episode, its arena_runtime.task_layer name.
string task_layer
# Axis -> comma-separated task modes.
diagnostic_msgs/KeyValue[] modes
# Robot model names active at this episode's reset.
string[] robots
uint8 QUEUED  = 0
uint8 RUNNING = 1
uint8 SUCCESS = 2
uint8 FAILED  = 3
uint8 SKIPPED = 4
uint8 FATAL   = 5
uint8 outcome_state
string outcome_info
string goal_uuid
# Goal progress of the robot that closed the least of its start distance, metres. Zero when no GoTo goal was active.
float32 goal_dist_start
float32 goal_dist_min
float32 path_length
builtin_interfaces/Time start_time
# Why the episode is compromised, empty when intact
string[] integrity
# Effective params of the selected modes, full names, staged overlaid if queued.
rcl_interfaces/Parameter[] params
# Human backend params the active backend accepted, staged dict overlaid for queued records.
rcl_interfaces/Parameter[] human_params
# JSON list of {op, p, q, text} episode conditions, empty when none
string conditions
# JSON robot name -> {phases, conditions, map_poses, goal_inputs, instructions}
string phases
