# One active attention channel, resolved to joints by the animation layer

# Body part: head | arm (layer picks the side) | arm_l | arm_r | body
string slot

# Target in world frame (aimed slots)
geometry_msgs/Point at

# body: animation clip name
string clip

# arm: dominant hand l | r, empty = default
string hand
