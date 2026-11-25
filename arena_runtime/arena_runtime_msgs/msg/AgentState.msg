# One agent of a task layer and what it is doing
string name
# "robot" or "pedestrian"
string kind
# Current or last task, empty while idle
string task
# idle, assigned, working, met, failed or dropped
string state
# Zone refs the task goes through
string[] zones
# Sim seconds since the episode start of the last change
float64 since
