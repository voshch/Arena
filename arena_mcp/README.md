# arena_mcp

One MCP server for all of Arena. Every package publishes its tools as a domain on the shared core in `arena_mcp.core`, the server loads every installed domain, and `arena mcp` (in the container) or `_meta/tools/arena-mcp` (on the host) is the entry point.

`python3 -m arena_mcp ls` lists the installed domains, served or unavailable with the reason. `python3 -m arena_mcp ls <domain> [--json]` lists a domain's tools, and the resource `arena://domains` carries the same for a connected client. Tools are named `<domain>_<verb>`. Env-scoped tools take `env_id`, which may be omitted while exactly one env runs. Long operations return a job, followed with `jobs_status`, `jobs_wait`, `jobs_log` and `jobs_cancel`. Errors end with a `hint:` line naming the next step.

## Writing a domain

A domain lives in the package whose code it drives:

```python
from arena_mcp.core import Domain, ToolError, ROS, tg

domain = Domain("demo", """One line on what it covers.""", requires=("demo_msgs",))


@domain.tool(read_only=True)
async def demo_state(env_id: int | None = None) -> dict[str, object]:
    """What the agent reads in the tool list."""
    ...
```

- Register it in the package's `setup.py` (`entry_points={'arena_mcp.domains': ['demo = demo_pkg.mcp:domain']}`, or `setup.cfg` for ament_cmake packages that install Python through `ament_python_install_package`) and add `<exec_depend>arena_mcp</exec_depend>` to its `package.xml`. A package that is also an editable install in the venv lists `entry-points` under `dynamic` in its `pyproject.toml`, otherwise the venv's metadata hides the entry point.
- `requires` names importable modules. When one is missing the domain is listed as unavailable instead of failing the server, so keep heavy imports inside tool bodies.
- `@domain.tool(read_only, destructive, idempotent, open_world)` sets the MCP annotations. Only tools annotated to return a dict get a structured output schema, every other result (lists, images, audio, mixed content) goes out as plain content. Sync tools run in a worker thread, ROS calls belong in async tools.
- `ROS` is the server's one ROS node: service calls that fail fast when nothing serves the name and leave no client behind (a lingering client makes a name like `/arena/register_env` look live), action goals, latched-topic caches, one-call topic reads, publishers, parameters with type coercion, the graph and sim time. QoS names map to the `arena_rclpy_mixins.qos` profiles. A crashed executor fails every later call with the error instead of a timeout.
- `tg`, `env_ns`, `env_table`, `fleet` and `robot_ns` resolve env ids and robot names against the runtime's registry (`ENVS_TOPIC`). `SUBMIT_TASK_TIMEOUT_S` bounds a task generator's `runtime/submit_task`, which waits on the reset lock. `data_dir()` is `$ARENA_DATA_DIR`.
- `JOBS` and `cli.job` run long work as jobs: in-process tasks end with the server, process jobs run in their own session, outlive it and are listed again by the next server, which tells a reused pid apart by its start time (`job.alive`, the owning process for a task job). `cli` runs verbs through the `_arena_cli` function a sourced shell exports, in `$ARENA_WS_DIR`. `proc.stop_group` stops a process group with SIGINT first, then SIGTERM and SIGKILL after the grace windows of `arena_bringup.reaper`, and `JOBS.stopper` gives a job kind its own stop. A cancel hook's `ToolError` reaches the `jobs_cancel` caller and leaves the job running.
- `msgs` converts messages to and from plain dicts without dropping fields, `media` turns PNGs, image messages and occupancy grids into image content, `ToolError(message, hint)` is the error type.
- Tests go in a `mcp/` subdirectory of the package's tests with `pytest.importorskip("arena_mcp")`, and run the real tools against real files and real ROS nodes created in the test.

## Launcher and clients

The server speaks MCP over stdio and runs inside the arena container, where ROS, the workspace and the CLI live. An MCP client on the host reaches it through `_meta/tools/arena-mcp`:

```
_meta/tools/arena-mcp [--fork NAME] [--host SSH_HOST] [--container NAME] [serve] [DOMAIN ...]
_meta/tools/arena-mcp [...] ls [DOMAIN ...] [--json]
```

- It sources `_meta/docker/lib` for the workspace (and `ARENA_FORK` with `--fork`), finds the running compose service `arena` of its `ARENA_PROJECT_NAME` with `arena_containers`, and runs `python3 -m arena_mcp` there through `arena_docker exec -i <container> /entrypoint.sh` in a sourced shell. `--container` names the container instead. Without `ARENA_WS_DIR` the workspace is the one the script sits in (`<ws>/src/Arena`), else `~/arena_ws`.
- `--host` runs the lookup and the exec on another machine over `ssh -T -o BatchMode=yes`, for forks on a shared box, where `docker` may be rootless podman. The lib's functions and project name go along in the remote command.
- It never creates, starts or recreates a container. Without a running one it exits 1 and names the command that starts it (`source arena`, or `arena fork new NAME` in the dev container for a fork). A container still in its first-boot setup is refused, since the setup prints to stdout.
- Stdout carries only the protocol, every launcher message goes to stderr. `--print-command` prints the compose project and the lookup, probe and exec commands, one bash-quoted line each, instead of running them.

`.mcp.json` at the Arena root registers the launcher as the server `arena` with `serve`, for clients that read project configuration. Pass domain names after `serve` to serve only those.

Inside the container, `arena mcp` lists the installed domains, served or unavailable with the reason, and prints the registration line. `arena mcp ls [DOMAIN ...] [--json]` adds each named domain's tools, `arena mcp serve [DOMAIN ...]` serves them.

## Calling a tool from a terminal

`arena mcp call TOOL [KEY=VALUE ...]` (`python3 -m arena_mcp call`) runs one tool in its own short-lived process, through the same server code as `serve`, loading only the tool's domain. `arena mcp call TOOL --help` prints the tool's description and parameters, and `arena mcp call <TAB>` completes tool names and then their keys from the completion manifest.

```
arena mcp call worlds_preview name=map_empty
arena mcp call entities_spawn_static model=fixture_cabinet_a01 x=2 y=0
arena mcp call system_launch sim=dummy world=map_empty 'args={"task.robots": "explore"}'
```

- A string parameter takes the value as is, any other parses it as JSON and falls back to the text, and `null` clears an optional parameter. An unknown or repeated key, a missing required one, or an unknown tool exits 2 and names what exists.
- Stdout carries only the result: text as is, a dict as JSON, image and audio content saved under `$ARENA_DATA_DIR/mcp/<hostname>/media` with the path printed. Logs and tool errors go to stderr. The exit code is 0 on success, 1 on a tool error or a job that did not succeed, 130 after Ctrl-C.
- A job the tool starts runs in the foreground, in-process and process jobs alike, since nothing would watch it after the call exits. Its log streams to stderr and its progress follows once a second, its final status goes to stdout. Ctrl-C cancels it the way `jobs_cancel` does, and a second Ctrl-C exits at once and may leave it running.
- Other processes see the job: `arena mcp call jobs_status job_id=<id>` or `jobs_wait` from a second terminal, or a running server's `jobs_status`, read its record. A task job records its owning process, and counts as lost once that process is gone.
- Each call starts its own ROS node only if the tool touches ROS. That node first waits one DDS announcement period (`ROS.settle_discovery`), which adds about 3.5 s, so the call sees the whole graph. A fresh publisher waits for its subscribers to match, and for their acknowledgement before the process exits. What a server keeps between calls, such as a cached topic, lasts for that one call.

## system

The `system` domain runs the stack lifecycle through the arena CLI: `system_launch` (`arena launch`, envs and runtime), `system_runtime` (`arena runtime`), builds, rebuilds, tests, `arena deps` and `arena preload`, each as a process job that outlives the server.

- A launch passes its parameters and `args` as the KEY:=VALUE tokens `arena launch` takes, with the CLI's defaults: rviz starts unless `headless` (or `viz:=false` in `args`). `args` may not repeat a key the tool has a parameter for. Like `arena launch`, it adds its envs to a runtime that is already up, and like it, an unknown sim fails in the runtime, ending the job.
- `system_stop` sends SIGINT to the launch job's process group and waits for its whole process tree, recorded before the signal, by default for the supervisor's whole graceful stop (the `arena_bringup.reaper` windows). Survivors come back with their command lines, and each further call escalates: the `arena launch` supervisor turns a repeated SIGINT into SIGTERM and then SIGKILL for its children, and processes it left behind get SIGTERM, then SIGKILL. `jobs_cancel` on a launch job runs these steps to the end.
- `system_stacks` lists the running launch jobs and the outermost `ros2 launch`, `ros2 run` and supervisor processes this server did not start.
- `system_logs` tails a job log, or the newest ROS log of a node under rclpy's logging directory (`$ROS_LOG_DIR`, else `$ROS_HOME/log`, else `~/.ros/log`), found by file name or by the node's logger in the file.

## workspace

The `workspace` domain covers the CLI verbs that change the workspace or its containers: `workspace_status` (git state of the metarepo and every initialized submodule, installed features, net and gpu settings with their sources), `arena update`, `arena feature <name> install|update|stop|uninstall`, `arena settings net`, `arena feature docker gpu`, `arena fork ls|new|down|exec` and `arena feature robots ls|add|rm|update`. Feature and action names come from the CLI's feature modules. Every mutating tool is marked destructive, settings changes apply once `source arena` recreates the container, and the fork verbs refuse inside a fork with the CLI's own message.

## ros

The `ros` domain is the generic long tail for what no Arena domain covers: the graph with types, interface definitions, any service call, topic echo (images and occupancy grids as image content, long arrays clipped) and publish, node parameters, and any action goal as a job whose progress carries the feedback and whose cancel cancels the goal. Types are `package/kind/Name` or `package/Name`, and the graph supplies them when omitted.

`python3 -m arena_mcp ls system workspace ros` lists the tools.
