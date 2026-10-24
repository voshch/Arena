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

# Active upper-body attention channels, compose with animation_state
Gesture[] gestures

# Empty = synthesize fallback gait, non-empty = upstream override.
sensor_msgs/JointState joint_state

# Monotonic walk-cycle phase in rad, 0 = unknown.
float32 gait_phase

# Current cycle rate in Hz, 0 = unknown.
float32 gait_cadence

# Agent type, a builtin name or a yaml path, empty = adult.
string agent_type

# Resolved local SDF path for this ped's actor, read once per name by the spawner.
string model_uri

# Active humansim interaction, -1 = none.
int32 interaction_id -1
# Kind of that interaction, meaningful only when interaction_id >= 0.
uint8 interaction_type

# Height of the surface the ped rests on (seat cushion, mattress) while it holds a SIT_ON/LIE_ON interaction,
# m above pose.position.z (its floor). NO_SUPPORT = on its feet, or the surface is unknown.
float32 NO_SUPPORT = -1.0
float32 support_height -1.0
