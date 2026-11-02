# Standard pedestrian representation for Arena simulators

# Identification
string name
uint32 id

# Spatial information
# Body frame: origin at the soles under the pelvis, z up the body.
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

# Empty = fallback gait, else semantic override. On arena_peds: driven joints
sensor_msgs/JointState joint_state

# Skeleton pose on the Arena body rig, no joints = none
BodyPose body

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
