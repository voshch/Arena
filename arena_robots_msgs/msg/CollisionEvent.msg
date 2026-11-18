# One footprint contact, attributed to the entity hit

string KIND_WALL="wall"
string KIND_STATIC="static"
string KIND_PEDESTRIAN="pedestrian"
string KIND_DEVICE="device"

# one of the KIND_* constants
string kind
string obstacle_id
float32 distance
geometry_msgs/Point obstacle_position
# KIND_DEVICE: the kind of the held device touched (arena_people_msgs/HeldDevice), obstacle_id is its holder
string device
