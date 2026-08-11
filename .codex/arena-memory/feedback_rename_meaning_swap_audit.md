---
name: feedback-rename-meaning-swap-audit
description: "When a rename swaps a name's meaning (old type takes new name, new type takes old name), grep for the name doesn't verify correctness. Audit OLD-API usage patterns explicitly."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4ff60925-c6b9-407a-a92b-9dd0eb233ee2
---

When agent reports a rename pass, do NOT verify by grepping for the renamed-away identifier. If the rename swapped meanings (e.g., old `WorldDescription` became `FloorDescription`, and a different type took the name `WorldDescription`), the name still resolves; old-API usages compile and pass ruff. The bug only shows up at runtime.

**Why:** Burned during Agent A's restructure on PR #44 multi-level world. `arena_simulation_setup/utils/generative/{__init__,empty,hallway}.py` used `WorldDescription.Zone` and `WorldDescription(zones=...)` — old API. Both my grep sweep and ruff passed because `WorldDescription` is still a valid name (just bound to a different class). Test collection failed at runtime. User was rightfully angry.

**How to apply:** After any meaning-swap rename, audit by grepping for **OLD API patterns** the old type uniquely had, not for the renamed-away name:
- Nested classes that only existed on the old type (`OldType.NestedThing`)
- Constructor signatures unique to the old type (`OldType(specific_kwarg=...)`)
- Methods that don't exist on the new type
- Imports `from x import OldName` in modules outside the rename author's touched-file list

Or run a type checker (mypy/pyright). Greps and ruff can't catch this class of error.

Even cheaper: spot-read 3-5 random files in the same package the agent touched. If the agent reports "19 files modified" and you only ran greps, you skipped the verification step. Trust-but-verify means open files.

Related: [[feedback-rclpy-mixins-first]] for similar "look at what's actually there, don't assume" pattern.
