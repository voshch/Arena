---
name: Rename agent_template to agent
description: User wants agent_template renamed to just agent across msgs, types, scenario yaml, and adapter code
type: feedback
---

Rename `agent_template` to `agent` everywhere (msg fields, internal types, scenario yaml keys, adapter code).

**Why:** "template" is an implementation detail (it's a blueprint that gets sampled per-spawn). From the scenario author's perspective it's just "what kind of agent does this source produce." `agent` is cleaner.

**How to apply:** When executing this rename, touch: `AgentTemplate.msg` (consider renaming the msg type itself to `Agent` or `AgentConfig`), `SourceConfig.msg`, internal `SourceConfig.agent_template` field, `spawn_scheduler._sample_spawn_request`, `arena_humansim.py` adapter msg building, and all scenario yaml files (`agent_template:` key → `agent:`). Bundle with other pending renames if any.
