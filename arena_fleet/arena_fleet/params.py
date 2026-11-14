from arena_rclpy_mixins.params import Choices, Param, ParamGroup, Range


class FleetParams(ParamGroup):
    WORLD = Param[str]("world", "", "world.yaml of the Arena world the fleet runs in.")
    RESOURCES = Param[str]("resources", "", "Resource table with the managed doors (kind: door) of that world.")
    SCENARIO = Param[str]("scenario", "", "scenario.yaml whose robot starts park the fleet; the map offset is read from them.")
    OUT = Param[str]("out", "", "Directory for events.jsonl, state.jsonl, samples.jsonl, heat.jsonl and dispatch.json.")
    COST = Param[str]("cost", "", "Cost function YAML. Empty = the package's configs/cost.yaml.")
    RULES = Param[str]("rules", "", "Rules and profiles YAML. Empty = the package's configs/rules.yaml.")
    PROFILE = Param[str]("profile", "", "Profile of the rules file that switches rules on. Empty = its default_profile.")
    INFO = Param[str](
        "info",
        "none",
        "What the dispatcher knows about people: none, oracle (everyone), perceived (what robots see), flow (live state and map of dynamics), mod (map of dynamics only), dmod (live state following a directional map of dynamics, which also prices walking against the stream).",
        constraint=Choices("none", "oracle", "perceived", "flow", "mod", "dmod"),
    )
    MOD = Param[str]("mod", "", "Map of dynamics .npz (rho, x_edges, y_edges, t_edges) for info flow and mod.")
    LOCAL = Param[str]("local", "drlvo", "Local planner that follows the plan: drlvo (plan topic, global_planner none) or dwb (Nav2 FollowPath).", constraint=Choices("drlvo", "dwb"))
    ROBOT = Param[str]("robot", "jackal", "Robot model name; robots are <robot>_<i>, a single one <robot>.")
    N = Param[int]("n", 4, "Fleet size.", constraint=Range(1, None))
    NS = Param[str]("ns", "/arena/env_0", "Namespace of the Arena env.")
    SEED = Param[int]("seed", 0, "Seed of the goal draw.")
    DURATION_S = Param[float]("duration_s", 180.0, "Sim seconds after setup before the episode ends.", constraint=Range(0.0, None, lo_open=True))
    TICK_S = Param[float]("tick_s", 0.25, "Seconds between decisions.", constraint=Range(0.0, None, lo_open=True))
    SAMPLES = Param[bool]("samples", True, "Write input-output samples of every decision to samples.jsonl.")
