from __future__ import annotations

import asyncio

from arena_bringup.viz_backends import env_idx_from_ns

from arena_mcp.core import Domain, ToolError, proc, tg
from rerun_utils import views

domain = Domain(
    "viz",
    """Live web views of an env in the Rerun viewer: viz_rerun_start runs the rerun bridge of an env detached and returns the viewer URL,
    viz_rerun_list shows the views every process (this server, the CLI, the dashboard) started, viz_rerun_stop ends one.""",
    requires=("rerun", "rclpy"),
)

NO_VIEW = "viz_rerun_list shows the live views"


@domain.tool(idempotent=True)
async def viz_rerun_start(env_id: int | None = None, web_port: int = 0, grpc_port: int = 0) -> dict[str, object]:
    """Start the Rerun web view of an env and return its viewer URL, or the live view when the env has one (started false). A port of 0 picks a free one."""
    ns = await tg(env_id)
    env = env_idx_from_ns(ns) if env_id is None else env_id
    view = views.load(env)
    if view is not None:
        return {**_start_result(view), "started": False}
    try:
        view = await asyncio.to_thread(views.spawn, env, ns, web_port, grpc_port)
    except views.StartTimeout as e:
        await proc.stop_group(e.pid)
        raise ToolError(str(e), hint="system_logs shows the bridge log, the env must have finished its first reset") from e
    except RuntimeError as e:
        raise ToolError(str(e), hint="is the runtime up? the bridge waits for the env node") from e
    return {**_start_result(view), "started": True}


@domain.tool(destructive=True, idempotent=True)
async def viz_rerun_stop(env_id: int | None = None) -> dict[str, object]:
    """Stop the Rerun web view of an env: SIGINT to the bridge's process group, escalating after the grace period."""
    if env_id is None:
        live = views.load_all()
        if len(live) != 1:
            raise ToolError(f"{len(live)} rerun views are up, pass env_id", hint=NO_VIEW)
        env_id = live[0].env_id
    view = views.load(env_id)
    if view is None:
        return {"env_id": env_id, "stopped": False}
    await proc.stop_group(view.pid)
    views.forget(env_id)
    return {"env_id": env_id, "stopped": True}


@domain.tool(read_only=True)
def viz_rerun_list() -> dict[str, object]:
    """The live Rerun views of this machine: env, bridge pid, ports and viewer URL."""
    return {"views": [view.public() for view in views.load_all()]}


def _start_result(view: views.View) -> dict[str, object]:
    public = view.public()
    del public["alive"]
    return public
