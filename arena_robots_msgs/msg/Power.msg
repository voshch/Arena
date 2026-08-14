std_msgs/Header header # stamp/frame copied from the driving joint_states message
float32 total_power_w # static plus mechanical plus thermal power
float32 static_power_w # constant compute, idle motor and component draw
float32 total_mechanical_power_w # sum of per-joint mechanical power
float32 total_thermal_power_w # sum of per-joint resistive losses
string[] joint_names # joint order for the per-joint arrays below
float32[] joint_mechanical_power_w # per-joint |effort*velocity| over drivetrain efficiency
float32[] joint_thermal_power_w # per-joint heating coefficient times effort squared
float32[] joint_total_power_w # per-joint mechanical plus thermal power
