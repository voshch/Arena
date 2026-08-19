# One upper-body attention channel, resolved to joints by the animation layer

# Body part: head | arm (layer picks the side) | arm_l | arm_r
string slot

# Target in world frame
geometry_msgs/Point at

# JSON options for the slot ({"dominant": "l|r"}), empty = defaults
string opts
