---
name: feedback-deslop-docstrings
description: Docstrings are one terse load-bearing line; no rationale paragraphs; deslop verbose docstrings in passing
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5ba49527-47d6-4a50-9f6e-500945128630
  modified: 2026-08-01T13:14:50.746Z
---

Module/class/function docstrings are ONE short line stating the load-bearing fact only. No rationale narration (testability justifications, design-why paragraphs), no restating the signature or return tuple in prose, no marketing phrasing ("GUI-authored pedestrian truth"). When editing a file that carries a verbose docstring, tighten it in passing.

NO COMPARATIVE CLAUSES. Never justify a symbol by contrast with its sibling or its predecessor: "exactly (unlike the float `to_seconds`)", "not a field-wise disjunction", "so the float path loses it". A docstring states what the thing guarantees, timelessly. `to_nanoseconds` gets "Convert to nanoseconds", full stop, matching the sibling's phrasing. A constraint IS load-bearing and stays ("nanosec is uint32 on the wire, so it normalizes into [0, 1e9)").

**Why:** User on 2026-07-18 reviewing the human_steering/manual implementation: "sloppy docstring, way too verbose. deslop as you go." Sharpened 2026-08-01 on the Time cleanup: "stfu with 'unlike the'. useless tombstone."

**How to apply:** Same bar as [[feedback-match-comment-density]] and [[feedback-useless-comments-scope]], extended to docstrings; agent prompts for this repo must carry the directive verbatim.
