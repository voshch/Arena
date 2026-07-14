# Authoring a new VLA/VLN model

End-to-end guide for serving a new vision-language navigation model through the
`vla_server` feature. Integration means writing a small HTTP **runner** that the
daemon launches on demand and that emits the actions contract; the Arena side
(the `vla` mobile adapter) needs no changes beyond pointing at your model.

The reference runner is [server_omnivla_edge.py](server_omnivla_edge.py); copy it
as a starting point.

## Prerequisites

- The feature installed once: `arena feature vla_server install`.
- Your weights published somewhere `curl` can fetch (HuggingFace today).

## 1. Add a `models.yaml` row

[models.yaml](models.yaml) is the model registry. Add one row keyed by the model
name you will select:

```yaml
my-vln:
  repo: Org/my-vln        # HuggingFace repo id
  weights: my-vln.pth     # file in the repo; also the file your runner loads
  server: server_my_vln.py  # runner script, next to this file
```

## 2. Write the runner (`server_my_vln.py`)

The daemon launches it as `python3 server_my_vln.py --port <p> --weights <path>`,
so it must accept `--port` and `--weights` and serve these endpoints:

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | (none) | `200` once the model is loaded and ready |
| POST | `/reset` | form: `session` | drops that session's history |
| POST | `/act` | multipart: file `image` (JPEG) + form `instruction`, `session` | the actions JSON below |

`session` is a per-robot id (the robot's namespace); keep any rolling observation
history keyed by it, and clear it on `/reset`. The daemon loads each model once
and caches it, so a single runner serves all robots/envs requesting that model;
concurrent robots serialize on one GPU.

The `/act` response is the wire contract; its typed schema in
[contract.py](../../../../task_generator/task_generator/tasks/robots/vla/contract.py)
is the SSOT:

```json
{
  "actions": { "mobile": { "waypoints": [
    { "x": 0.5, "y": 0.0, "yaw": 0.1 },
    { "x": 1.0, "y": 0.2, "yaw": 0.2 }
  ] } },
  "meta": { "intent": "heading for the doorway" }
}
```

- `actions.mobile` is a key-tagged union; emit **exactly one** form. `waypoints`
  is the only form consumed today (base-relative, meters and radians).
- `meta` is optional; omit it (or `meta.intent`) if your model emits no language.
  When present, `intent` is shown as a text marker above the robot.

## 3. Download weights and build

```bash
arena feature vla_server update
```

`update` fetches every `models.yaml` row's weights into
`$ARENA_DATA_DIR/vla/<model>/<weights>` (idempotent, skips existing) and rebuilds
the image so your runner script is baked in.

## 4. Select your model

Model selection is a constant today: set `_MODEL` in
[adapters/mobile/vla.py](../../../../task_generator/task_generator/tasks/robots/adapters/mobile/vla.py)
to your `models.yaml` key (`my-vln`). It becomes a registry-backed param once
multi-model selection lands.

Then run a VLA episode (`tm_robots:=vla`); the adapter ensures the daemon, calls
`POST /ensure {model}` to load your runner, and connects for `/act`.

## Adding a new action form

To return something other than `waypoints` (for example direct `cmd_vel`):

1. Add the typed variant to the `mobile` union in [contract.py](../../../../task_generator/task_generator/tasks/robots/vla/contract.py) and its `_parse_mobile` branch.
2. Register a handler for it in the `_mobile_handlers` map in [adapters/mobile/vla.py](../../../../task_generator/task_generator/tasks/robots/adapters/mobile/vla.py).

The adapter dispatches per form, so existing models keep working unchanged.
