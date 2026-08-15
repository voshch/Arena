from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mcp.types import (
    ReadResourceRequestParams,
    ReadResourceResult,
    Resource,
    TextResourceContents,
)

from .tools import _param_value_to_python

if TYPE_CHECKING:
    from .ros_bridge import RosBridge


def _record_to_dict(record: object) -> dict:
    return {
        "episode_id": record.episode_id,
        "world": record.world,
        "seed": record.seed,
        "tm_robots": record.tm_robots,
        "tm_obstacles": record.tm_obstacles,
        "tm_modules": list(record.tm_modules),
        "outcome_state": record.outcome_state,
        "outcome_info": record.outcome_info,
        "goal_uuid": record.goal_uuid,
        "integrity": record.integrity,
        "obstacles_params": [{"name": p.name, "type": p.value.type, "value": _param_value_to_python(p.value)} for p in record.obstacles_params],
        "robots_params": [{"name": p.name, "type": p.value.type, "value": _param_value_to_python(p.value)} for p in record.robots_params],
    }


def build_resources_list() -> list[Resource]:
    """Build the resource list served on resources/list."""
    return [
        Resource(
            uri="task_generator://state/world",
            name="Current world id",
            description="The world id currently active in the task_generator node. Updated whenever the world changes at a reset boundary.",
            mimeType="text/plain",
        ),
        Resource(
            uri="task_generator://state/episode",
            name="Episode state",
            description="Episode state as JSON: current episode record and queued (next) episode record.",
            mimeType="application/json",
        ),
    ]


def read_resource_content(params: ReadResourceRequestParams, bridge: RosBridge) -> ReadResourceResult:
    """Serve resources/read from the bridge state caches."""
    uri = str(params.uri)

    if uri == "task_generator://state/world":
        return ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri=params.uri,
                    mime_type="text/plain",
                    text=bridge.state_world,
                )
            ]
        )

    if uri == "task_generator://state/episode":
        cur = bridge.current_episode
        que = bridge.queued_episode
        payload = {
            "current": _record_to_dict(cur) if cur is not None else None,
            "queued": _record_to_dict(que) if que is not None else None,
        }
    else:
        payload = {"error": f"unknown resource: {uri}"}

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                uri=params.uri,
                mime_type="application/json",
                text=json.dumps(payload, default=str),
            )
        ]
    )
