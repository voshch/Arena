---
name: feedback_no_speculative_scope
description: "build exactly the requested behavior; don't bundle speculative convenience flags/features, even ones in an approved plan"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: dc9ecfee-81f1-47b0-966e-000798f2b15e
---

Implement the core ask and nothing more. Don't add convenience flags, affordances, or extra features I wasn't asked for, and don't speculatively build pieces the user signaled could wait.

**Why:** the user trims unrequested scope on sight even after nominally approving a plan that contained it. This session: cut `--at` and `--now` from the `arena robot` CLI ("didn't ask for that, don't need it"; placement is the task mode's job, resets happen in their own flow), and wanted the Fleet rviz panel deferred but I built it after an ambiguous go-ahead. A plan "go" is not blanket license for every flag in it.

**How to apply:** ship the minimal thing that satisfies the request. When I think of a convenience flag/feature/affordance, list it as an optional follow-up for the user to opt into, not baked into the diff. When a directive is bundled ("do X" + "also Y could be separate"), confirm sequencing before building the deferred part. Relates to [[feedback_planning_no_implementation]], [[feedback_questions_not_tasks]].
