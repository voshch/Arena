---
name: project-integrate-planner-skill
description: "the integrate-planner skill exists at .claude/skills/ — scoped to learned/DRL planners, deliberately local-only (gitignored), don't re-create or commit it"
metadata: 
  node_type: memory
  type: project
  originSessionId: c83ced68-0743-4e9b-a405-9a300932ea71
---

The `integrate-planner` skill (`.claude/skills/integrate-planner/SKILL.md`) is built and live (auto-loads). It is the full runbook for adding a **learned/DRL planner**: Phase 1 leaf commit, Phase 2 publish weights to the `arena-rosnav` HF org, Phase 3 host code under the `arena-planners` GitHub org (public, default branch `jazzy`), Phase 4 register the submodule in `arena_planners` (stanza carries `branch = jazzy` + `update = rebase`), Phase 5 bump the `Arena` pin.

Scope is learned planners only (`planner.yaml` + `weights.yaml`, edge-bridge). Classical/Nav2 planners (e.g. cohan / CoHAN-Nav2) have no `weights.yaml`, no HF step, and follow the nav2 path — explicitly out of scope. A `weights.yaml` with `files: []` (sicnav, crowdnav) is still in scope but skips Phase 2.

It was adversarially reviewed and the fixes were load-bearing (scope's shared-repo dependency, `files: []` planners, attngraph's multi-file/renamed weights would each have broken the first draft); walked against all 8 current learned planners and would have worked on each.

**Deliberately local-only.** `.claude/` is in the user's global `~/.gitignore`, so the skill is not tracked in the Arena repo. The user chose to keep it local (a personal runbook, not shared with the team) — do not force-add or commit it, and do not propose re-creating it. To share it would need `git add -f`. See [[project-planner-hf-weights-mechanics]] for the publish mechanics it encodes.
