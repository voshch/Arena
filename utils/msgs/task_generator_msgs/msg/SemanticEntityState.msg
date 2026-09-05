# Semantics for one annotated entity within an env snapshot.
string entity                # env-prefixed instance name (sim_path), e.g. "env_0/1_elevator"
string kind                  # semantic kind: "door" | "elevator" | "signal" | "schedule" | "gate" | "pressure_plate" | "occupancy_cap" | "sound"
string[] discrete_names      # discrete state names, index-aligned with discrete_values
string[] discrete_values     # discrete state token values
string[] continuous_names    # continuous state names, index-aligned with continuous_values
float64[] continuous_values  # continuous state float values
string[] predicate_names     # predicate names, index-aligned with predicate_values
bool[] predicate_values      # predicate boolean values
string[] members             # committed occupant/inside ids, empty when identity is not cheaply known
