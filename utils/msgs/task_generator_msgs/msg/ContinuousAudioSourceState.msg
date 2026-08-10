std_msgs/Header header

string source_id
int32 source_agent_id
string source_agent_name
string source_model
string sound_type
string source_backend

geometry_msgs/Point source_position
float32 source_yaw

float32 linear_velocity_mps
float32 angular_velocity_radps
float32 left_velocity_mps
float32 right_velocity_mps

float32 source_volume_db
float32 reference_distance_m
bool active
uint64 deterministic_seed

# Reserved for source models calibrated against real load telemetry.
bool has_normalized_load
float32 normalized_load

string[] semantic_tags
