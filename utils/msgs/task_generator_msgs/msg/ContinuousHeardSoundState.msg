std_msgs/Header header

string source_id
string listener_id
int32 source_agent_id
string source_agent_name
string source_model
string sound_type
string source_backend
string system_id
string asset_id
string label
bool loop
builtin_interfaces/Time program_start_time

geometry_msgs/Point source_position
geometry_msgs/Point listener_position

float32 linear_velocity_mps
float32 angular_velocity_radps
float32 left_velocity_mps
float32 right_velocity_mps

float32 source_volume_db
float32 received_volume_db
float32 direct_delay_sec
bool active
bool audible
bool occluded

string source_zone
string listener_zone
string propagation_backend
bool used_backend_fallback
string backend_fallback_reason

string[] portal_ids
string[] traversed_zones
geometry_msgs/Point[] portal_positions
uint8 portal_hop_count
float32 portal_route_loss_db

uint64 deterministic_seed
