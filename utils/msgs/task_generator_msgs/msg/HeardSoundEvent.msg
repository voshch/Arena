std_msgs/Header header
string event_id
string listener_id
int32 source_agent_id
string source_agent_name

string sound_type
string label
string asset_id

geometry_msgs/Point source_position
geometry_msgs/Point listener_position
float32 distance
float32 bearing_rad
float32 source_volume_db
float32 received_volume_db
float32 hearing_threshold_db
float32 direct_delay_sec
bool audible
bool occluded
uint8 propagation_level
task_generator_msgs/AcousticPath[] early_paths
float32 reverb_rt60_sec
float32 reverb_gain_db
string source_zone
string listener_zone

# The model level is retained above for compatibility. These fields report
# the backend that actually produced this event and any fallback decision.
string propagation_backend
bool used_backend_fallback
string backend_fallback_reason
string portal_id
geometry_msgs/Point portal_position

# Complete portal route. portal_id/portal_position above remain the first
# portal for compatibility with existing consumers.
string[] portal_ids
string[] traversed_zones
geometry_msgs/Point[] portal_positions
uint8 portal_hop_count
float32 portal_route_loss_db
