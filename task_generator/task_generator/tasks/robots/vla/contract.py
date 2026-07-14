"""Typed schema for the vla_server `/act` response, the contract SSOT.

`actions` maps a robot cap to a single action. The cap key (`mobile`, and later `arm`, `gripper`,
`lift`) selects the adapter that realizes it, and a mobile manipulator returns several caps at once.
Each cap value is a disjoint union keyed by action form: the present key is the tag and gates the
payload. Conventions fixed here, not repeated on the wire: mobile `waypoints` are robot-base-relative
in meters and radians, and timed forms carry their own `dt`.

The vla_server emits the plain-dict mirror of these types, and `parse` structures it back.
"""

import attrs


@attrs.frozen
class Waypoint:
    x: float
    y: float
    yaw: float


@attrs.frozen
class Waypoints:
    """Spatial path, base-relative (m, rad). No timing. (omnivla-edge emits this.)"""

    steps: list[Waypoint]


# mobile cap union: Waypoints today, cmd_vel (direct velocity) is the other anticipated form.
MobileAction = Waypoints


@attrs.frozen
class Actions:
    mobile: MobileAction | None = None


@attrs.frozen
class Meta:
    intent: str = ""


@attrs.frozen
class Response:
    actions: Actions
    meta: Meta


def _parse_mobile(form: str, value: object) -> MobileAction:
    if form == "waypoints":
        return Waypoints([Waypoint(float(s["x"]), float(s["y"]), float(s["yaw"])) for s in value])
    raise ValueError(f"unknown mobile action form: {form!r}")


def parse(payload: dict) -> Response:
    """Structure the `/act` JSON body into a typed response, dispatching each cap's key-tagged union."""
    actions = payload["actions"]
    mobile = None
    if (cap := actions.get("mobile")) is not None:
        ((form, value),) = cap.items()  # exactly one key per cap union
        mobile = _parse_mobile(form, value)
    meta = payload.get("meta") or {}
    return Response(actions=Actions(mobile=mobile), meta=Meta(intent=str(meta.get("intent", ""))))
