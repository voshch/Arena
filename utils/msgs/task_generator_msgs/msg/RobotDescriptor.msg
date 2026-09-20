# Per-robot identity for an env's fleet.
string name        # instance name (matches legacy robot_names entry)
string model       # robot model id, resolvable via RobotIdentifier
string ns          # ROS namespace, absolute (leading slash)
string frame       # TF frame prefix, no leading or trailing slash
string base_frame  # TF frame id of the robot base (e.g. env_0/jackal/base_link)
