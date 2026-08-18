# Standard pedestrian representation for Arena simulators

# Identification
string name
uint32 id

# Spatial information
geometry_msgs/Pose pose
geometry_msgs/Twist twist

# Animation/behavior states (integrated)
uint8 IDLE = 0
uint8 WALKING = 1  
uint8 RUNNING = 2
uint8 PANIC = 3
uint8 SURPRISED = 4
uint8 CURIOUS = 5
uint8 THREATENING = 6
uint8 animation_state

# Upper-body gesture, composes with animation_state
Gesture gesture

# Empty = synthesize fallback gait, non-empty = upstream override.
sensor_msgs/JointState joint_state

# Monotonic walk-cycle phase in rad, 0 = unknown.
float32 gait_phase

# Resolved local SDF path for this ped's actor, read once per name by the spawner.
string model_uri

