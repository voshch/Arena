# One resolved, effective cap on a live robot.
string cap        # "mobile" | "arm" | ...
string adapter    # bound driver kind: "nav2" | "moveit" | "none"
string instance   # mount name the instance landed on; empty for singletons
string variant    # morphology value that enabled it; empty if not variant-gated
