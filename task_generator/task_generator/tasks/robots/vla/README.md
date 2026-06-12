# VLA navigation task mode (`TM_VLA`)

## Changes

Each cycle (throttled to `_INFERENCE_INTERVAL`):

1. **Image subscription is automatic** — on reset the task mode finds each robot's camera topic by
   itself, preferring any sensor with `"front"` in the name.
2. The latest frame is sent to the inference server over HTTP, along with the instruction.
3. The model runs inference and returns **8 future waypoints**, sent back to the client.
4. The waypoints are filtered through `is_valid_pose` (anything off-map or inside an obstacle is
   dropped), and the **furthest valid one** is submitted as the goal.
5. If the total distance from the current pose to that goal is below
   `_WAYPOINT_PROXIMITY_THRESHOLD`, control is **handed off to nav2** to finish the approach.

## Server

Adds the `vla_server` arena feature.

Run with:

```bash
arena feature vla_server install
arena feature vla_server update
```

```bash
arena vla_server:=omnivla_edge
```

`omnivla_edge` is currently the only supported model.

Download model with
```bash
git clone https://huggingface.co/NHirose/omnivla-edge
```
put model.pth in /arena_ws/src/Arena/_meta/docker/features/vla_server/model folder
## Note

The instruction is set in `impl.py` for now and has to be edited there directly — this will move to
a proper arg later.