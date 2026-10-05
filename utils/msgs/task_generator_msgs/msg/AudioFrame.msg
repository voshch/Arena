std_msgs/Header header

# Interleaved IEEE-754 float PCM.  For frame f and channel c the sample is
# data[f * channel_count + c].
uint32 sample_rate
uint16 channel_count
uint32 frame_count
string encoding
bool interleaved

# Stable semantic metadata. Arrays are either empty or channel_count long.
string[] channel_names
string[] frame_ids
geometry_msgs/Point[] microphone_positions
float32[] microphone_yaw_rad

float32[] data
