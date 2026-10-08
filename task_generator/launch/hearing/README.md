# Robot hearing dispatch

Entry point: [`hearing.launch.py`](hearing.launch.py).

Called from `task_generator.launch.py`'s OpaqueFunction (once per
environment, only when `robot.hearing` is not `none`, which needs
`acoustics:=arena`) with `frontend` (the `robot.hearing` key), `policy`
(`robot.hearing.policy`), `namespace` and `environment_namespace`. Every
launch configuration of the parent stays visible to the backend, so
undeclared `robot.hearing.*` keys reach it by prefix.

## SelectAction dispatch

| Key | Action |
| --- | --- |
| `none` | empty group, no nodes |
| `bus`, `srp`, `seld` | includes [`arena/arena.launch.py`](arena/arena.launch.py) |

## arena backend

[`arena/arena.launch.py`](arena/arena.launch.py) fails with the
`arena feature hearing install` hint when the `arena_hearing` package is
missing. Otherwise it includes `arena_hearing`'s `launch/hearing.launch.py`
in isolation with:

| Arg | Value |
| --- | --- |
| `robot.hearing.<key>` | every non-empty one of the parent except `robot.hearing.policy`, the hearing launch strips the prefix |
| `env.ns` | absolute env namespace, the hearing nodes live in it |
| `tg_node` | task generator node name, the auditory topics live below it |
| `frontend` | `robot.hearing` |
| `policy` | `robot.hearing.policy` |

`robot.hearing.*` keys never reach the task generator node or the robot
adapters. The Nav2 speed-filter overlay comes from
`task_generator.simulators.hearing.nav2_overlay(hearing)`
(`arena_hearing/config/nav2_overlay.yaml`, empty for `none`) and lands as
`robot.mobile.params_overlay` unless that is set explicitly.

## Node side

`task_generator/simulators/hearing/` holds `HearingRegistry`, keyed
`none|bus|srp|seld`, yielding a `BaseHearing`: `robot_displays(robot)` (the
robot's `_hearing` manifest entry: belief grid, speed mask, wedges) and
`recorded_topics()`. `simulators/hearing/arena/` is the only task_generator
code importing `arena_hearing`.

## Layout

```
launch/hearing/
|-- hearing.launch.py      dispatcher
`-- arena/
    `-- arena.launch.py    install check, shim into arena_hearing's hearing.launch.py
```
