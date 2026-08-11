---
name: isaacsim-removed-locally
description: "run_isaacsim.py deletion in arena_isaac is intentional (user removed the isaacsim feature locally) — don't re-flag dangling refs"
metadata: 
  node_type: memory
  type: project
  originSessionId: c14a68a6-25f9-4d31-8702-c83624877f2a
  modified: 2026-08-05T04:30:40.111Z
---

As of 2026-08-05 the user intentionally deleted `arena_isaac/arena_isaac/run_isaacsim.py` in their local working tree ("removed the isaacsim feature locally, nbd"). The still-dangling references (setup.py console script, launch/run_isaacsim.launch.py, arena CLI isaac.py) are known and accepted. Don't re-flag this as a broken tree in reviews while the local state persists.
