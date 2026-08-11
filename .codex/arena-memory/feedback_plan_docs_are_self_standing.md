---
name: feedback_plan_docs_are_self_standing
description: "A plan document contains only the plan; decision tables, open questions, and revision history belong in chat, not in the artifact"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e84d971d-c58d-4e1e-bb07-e71a3766d80a
  modified: 2026-08-03T03:55:29.750Z
---

When the user asks for "a plan" or "the full plan", the document contains only
the plan: goal, background, requirements, design, scope limits, implementation
steps, tests. Keep out decision tables, open questions, "say go and I will
start", revision numbers, and any framing that corrects or references the
conversation ("revision 1 got this wrong", "corrections to your premise").

**Why:** the user rejected a plan doc that ended in a D1-D9 decision table with
"by plan i mean just plan". A plan has to read as self-standing to someone who
was not in the conversation. Decisions are a separate conversational artifact
with a different lifetime: they get resolved and then disappear into the design.

**How to apply:** once a decision is settled, state it in the design as a plain
fact, not as a choice with alternatives and rationale. Where a non-obvious
constraint drove the design, encode it as a numbered requirement the design
sections reference, so the reasoning survives without narrating the debate.
Track open questions and decisions needing input in chat instead. Relates to
[[feedback_text_deliverables_inline]], [[feedback_planning_no_implementation]],
[[feedback_no_justifying_comments]].
