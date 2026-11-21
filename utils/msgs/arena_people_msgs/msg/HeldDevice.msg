# Assistive device a pedestrian holds, projected on the ground in the frame of the pedestrian's pose

string CANE=cane
string ROLLATOR=rollator
string WALKING_FRAME=walking_frame
string IV_STAND=iv_stand

# One of the constants above, empty = holds none
string kind
geometry_msgs/Point32[] footprint
