# Same id may appear repeatedly across publications as outcome resolves;
# subscribers dedup by episode_id and keep the latest version per id.
uint32 episode_id
string world
int64 seed
string tm_robots
string tm_obstacles
string[] tm_modules
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
builtin_interfaces/Time start_time
# False = manually mutated mid-episode.
bool integrity
# Effective params for tm_obstacles only; live values overlaid with staged dict for queued records.
rcl_interfaces/Parameter[] obstacles_params
# Effective params for tm_robots only; live values overlaid with staged dict for queued records.
rcl_interfaces/Parameter[] robots_params
# JSON list of {op, p, q, text} episode conditions, empty when none
string conditions
