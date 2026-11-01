# One pose of the Arena body rig (task_generator human BODY.md)

# Joint names of the rig, any subset, a missing joint keeps its stance
string[] joints
# Rotation of each joint relative to its parent, in joints order
geometry_msgs/Quaternion[] rotations
# Root joint displacement from rest, body frame, m
geometry_msgs/Vector3 root_offset
