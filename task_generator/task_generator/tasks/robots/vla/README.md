# VLA navigation (`TM_VLA` + `vla` mobile adapter)

VLA navigation is split across two layers:

- **`TM_VLA`** ([impl.py](impl.py)) is a thin task mode. On reset it reads the `instruction` ROS param,
  picks random valid spawn poses, and submits one `VLAPhase(instruction=...)` per robot.
- **The `vla` mobile adapter** ([../adapters/mobile/vla.py](../adapters/mobile/vla.py)) is the central
  bridge. It subclasses the nav2 adapter, so robots run with `mobile:=vla`. It owns the vla_server
  lifecycle, the per-robot inference loop, the per-action-type handlers, and the intent visualization.

## Adapter cycle

Each cycle (every `_INFERENCE_INTERVAL`), per robot:

1. The latest camera frame (auto-discovered on reset, preferring a sensor with `"front"` in the name)
   is sent to the inference server over HTTP along with the instruction.
2. The server returns an `actions` envelope; a handler is dispatched per action form. For `waypoints`
   (omnivla-edge), the waypoints are filtered through the world occupancy grid and the furthest valid
   one is followed via the inherited nav2 stack.
3. When the distance to that goal drops below `_WAYPOINT_PROXIMITY_THRESHOLD`, the final goal is handed
   to nav2 to finish the approach and the phase completes once nav2 reports done.

`meta.intent` (optional on the wire, empty for omnivla-edge) is published as a `TEXT_VIEW_FACING` marker
above the robot on `<ns>/vla/intent` for any visualizer.

## Server

Adds the `vla_server` arena feature.

```bash
arena feature vla_server install
arena feature vla_server launch [<model>]
```

`install` downloads the weights listed in `models.yaml` into `$ARENA_DATA_DIR/vla/<model>/` and builds
the image. `launch` brings the thin **daemon** up detached (`docker compose up -d --wait`) on
`127.0.0.1:8000` and prints its port; it is idempotent (a no-op when already healthy) and holds no
model itself.

The adapter runs `arena feature vla_server launch` in `ensure_services` (the arena container has the
docker socket), then `POST /ensure {model}` to the daemon. The daemon spawns that model's runner (its
`server` script plus weights from `models.yaml`) on a free port if one is not already running and
returns the port; the adapter connects there for `/act`. Models are loaded on demand and cached, so
different envs/robots can request different models and each loads once. The launch surfaces docker
stderr only on failure; the success path is silent. An episode aborts if the daemon or the requested
runner fails to come up.

Per-robot isolation lives in the request, not the socket: each `/act` carries a `session` id (the
robot's env-prefixed namespace), and the server keeps a separate rolling observation history per
session. The adapter clears its session via `/reset` at the start of each episode. Concurrent robots
serialize on a single GPU; multiple GPUs would require a per-GPU runner pool (not yet implemented).

The `/act` response is a cap-keyed `actions` envelope (disjoint union per cap) plus an optional `meta`
block; its typed schema in [contract.py](contract.py) is the SSOT.

To serve a different VLA/VLN model, see
[AUTHORING.md](../../../../../_meta/docker/features/vla_server/AUTHORING.md).

## Note

The instruction is a ROS param on the `vla` mode (`instruction`), set via launch or the task config,
defaulting to the constant in `impl.py`. The model is selected by the `_MODEL` constant in the adapter
([../adapters/mobile/vla.py](../adapters/mobile/vla.py)) while there is one model; it becomes a
registry-backed (dropdown) param when multiple models are supported.
