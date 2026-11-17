# The assistive device a pedestrian holds, posed from its rendered gait phase, in the frame of the pedestrian's pose

string CANE=cane
string ROLLATOR=rollator
string WALKING_FRAME=walking_frame
string IV_STAND=iv_stand

# One of the constants above, empty = the pedestrian holds none
string kind
# Object model renderers draw for it (an Arena asset name), empty = none
string model
# cane: the fist's grip centre; rollator, walking_frame, iv_stand: the origin under the grips (z: its lift off the floor)
geometry_msgs/Point start
# cane: the tip; the others: = start
geometry_msgs/Point end
# The footprint's centre on the floor (cane: = end) and its heading (cane: the shaft's, start to end)
geometry_msgs/Point centre
float64 yaw
# cane: the shaft's radius; the others: 0
float64 radius
# rollator, walking_frame, iv_stand: footprint length along yaw, width, height; cane: 0
geometry_msgs/Vector3 size
