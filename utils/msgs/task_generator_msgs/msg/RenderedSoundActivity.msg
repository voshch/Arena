std_msgs/Header header

# The raw-array sample clock is authoritative. Discrete records have a
# half-open [start,end) interval. Continuous records are active/inactive state
# transitions at start_sample_index/start_time; end fields equal the start.
string stream_id
string event_id
string source_id
int32 source_agent_id
string source_agent_name
string source_type
string sound_type
string asset_id
string channel_name

bool continuous
bool active
uint64 start_sample_index
uint64 end_sample_index
builtin_interfaces/Time start_time
builtin_interfaces/Time end_time
