---
name: gui-diagnostics-carb-log
description: "Script Editor diagnostic snippets must output via carb.log_warn, not print, so the user can copy-paste results"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 26189c30-8f48-467e-852a-a6dc9b883705
---

When handing the user a diagnostic snippet for the Isaac GUI Script Editor, emit results with carb.log_warn (one line per fact), never print.

**Why:** print output lands in the Script Editor panel which the user cannot copy-paste from (they screenshot it, lines get cut). carb.log_warn lands in the console where text is selectable, and in any captured log file.

**How to apply:** build the snippet as `import carb` + `carb.log_warn(f"...")` per line, related: [[carb-logging-for-debug]] (same rule for arena_isaac code).
