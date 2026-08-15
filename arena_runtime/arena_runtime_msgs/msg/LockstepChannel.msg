string name        # display name
string topic       # topic to watch, may contain {env} placeholder
string type        # message type, e.g. arena_people_msgs/msg/Pedestrians
float64 period_s   # producer period in sim seconds
bool hard          # freeze-and-wait gate, false = observe only
