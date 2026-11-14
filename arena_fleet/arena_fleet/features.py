import math
import typing
from collections.abc import Sequence

from arena_fleet.geometry import Point

if typing.TYPE_CHECKING:
    from arena_fleet.dispatcher import FleetDispatcher

TAUS = (0, 2, 4, 6, 8, 10)
ACTION = {"approach": 0, "cross": 0, "to_goal": 0, "to_slot": 1, "queued": 1, "hold": 1, "yield": 2}
ROBOT_FEATURES = ("t", "n_robots", "x", "y", "cos_yaw", "sin_yaw", "speed", "goal_dx", "goal_dy", "goal_dist", "passed_door", "near_robot_m", "near_person_m")
DOOR_FEATURES = ("dist", "T", "t_door", "H", "people_at_arrival", "people_in_door_3s", "ttc_plan", "robots_heading", "robots_inside", "robots_within_6m", "open", *(f"people_{t}s" for t in TAUS))
TTC_H = 10.0


def names(doors: Sequence[str]) -> list[str]:
    """Feature names in vector order; only inputs the dispatcher knows, none a consequence of the robot's own decisions."""
    return [*ROBOT_FEATURES, *(f"{d}_{k}" for d in sorted(doors) for k in DOOR_FEATURES)]


def ttc_plan(D: "FleetDispatcher", path: Sequence[Point]) -> float:
    """Seconds until the people map shows someone on the path, TTC_H when clear or unknown."""
    if not path or D.view is None:
        return TTC_H
    for px, py, t, _ in D.walk(path, TTC_H):
        if D.view.occupied(t, px, py):
            return t
    return TTC_H


def features(D: "FleetDispatcher", n: str, speed: float) -> list[float]:
    r = D.rob[n]
    x, y, th = D.pose[n]
    gx, gy = r.target
    others = [D.pose[m][:2] for m in D.names if m != n and m in D.pose and D.rob[m].phase != "done"]
    people = [(v[0], v[1]) for v in D.seen.values()]
    f = [D.t_rel(), float(len(D.names)), x, y, math.cos(th), math.sin(th), speed, gx - x, gy - y, math.dist((x, y), (gx, gy)), 1.0 if r.door_used else 0.0, min([math.dist((x, y), o) for o in others] or [30.0]), min([math.dist((x, y), p) for p in people] or [30.0])]
    cr = D.cost["facility"]["crowd_r_m"]
    for d in sorted(D.doors):
        cx, cy = D.doors[d].c
        terms, path = D.terms(n, r.target, d, around=False) if not r.door_used else (None, [])
        t = terms or {"T": 0.0, "t_door": 0.0, "H": 0.0, "people_at_door": 0.0}
        heading = sum(1 for m, o in D.rob.items() if m != n and o.door == d and o.phase not in ("done", "to_goal"))
        near = sum(1 for m in D.names if m != n and m in D.pose and D.rob[m].phase not in ("done", "to_goal") and math.dist(D.pose[m][:2], (cx, cy)) <= 6.0)
        leaf = D.door_open.get(d)
        f += [math.dist((x, y), (cx, cy)), t["T"], t["t_door"], t["H"], t["people_at_door"], 1.0 if D.people_in_door(d) else 0.0, ttc_plan(D, path), float(heading), float(len(D.inside.get(d, {}))), float(near), leaf.prog if leaf else 1.0]
        f += [D.view.count(tau, cx, cy, cr) if D.view else 0.0 for tau in TAUS]
    return [round(v, 3) for v in f]


def labels(D: "FleetDispatcher", n: str) -> dict[str, object]:
    """Decision after the tick: door index, action (0 go, 1 wait, 2 yield) and target relative to the robot."""
    r = D.rob[n]
    x, y, _ = D.pose[n]
    d = r.door or r.door_used
    if r.slot_xy:
        tx, ty = r.slot_xy
    elif r.aside:
        tx, ty = r.aside
    elif r.door and not r.door_used:
        tx, ty = D.doors[r.door].c
    else:
        tx, ty = r.target
    doors = sorted(D.doors)
    return {"door": doors.index(d) if d in doors else -1, "action": ACTION.get(r.phase, 0), "phase": r.phase, "target": [round(tx - x, 2), round(ty - y, 2)]}
