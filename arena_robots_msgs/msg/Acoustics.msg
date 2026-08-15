# validity_flags bits
uint32 FLAG_SCRUB_UNCLASSIFIED=1 # wheel sides not derivable from joint names, scrub term is a guess
uint32 FLAG_NO_EFFORT=2 # effort array absent, load term forced to zero
uint32 FLAG_NO_VELOCITY=4 # velocity array absent, speed term forced to zero

std_msgs/Header header # stamp/frame copied from the driving joint_states message
float32 total_level_af_dba # A-weighted fast-time-weighted total level at 1m
float32 total_level_zf_db # Z-weighted total level at 1m, NaN while the model is A-weighted only
float32 baseline_level_dba # stationary baseline contribution
float32 drivetrain_level_dba # drive plus scrub contribution
float32 uncertainty_1sigma_dba # 1-sigma spread of the model estimate
uint32 validity_flags # bitmask of the FLAG_* constants above
string operating_state # idle, driving or scrubbing
string calibration_status # provenance of the parametric profile, never a sensor calibration
