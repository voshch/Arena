---
name: spike-before-full-regen
description: Never full-regen the 96-cell roster after a pipeline change without a user-tested small spike first
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d1058176-4dfa-4e43-b9e3-c10599875612
---

After any arena_humans pipeline change, build a ~3-cell spike, copy it into the local Arena cache, and let the user live-test it BEFORE launching the full 96-cell regen.

**Why:** Full regens cost ~66 min each and static probes have repeatedly missed render-level defects (side-view sticks missed scissored legs, chord metrics missed mesh-frame lean). The user burned multiple regen cycles on generations that failed on first look. Established 2026-07-18 ("i want a 3 pedestrian spike that i can test before you waste any more of my time").

**How to apply:** `./docker.sh roster /recipes/roster.yaml -o /out/<spike_dir> --only <cell> <cell> <cell>` with cells matching the user's current launch command (e.g. casual_male_asian_young, casual_female_african_young + one suit), copy over `_assets/default/<Domain>/Human/` with fresh `.ttl`, report the test command, wait for user sign-off, only then full regen. See [[peds-all-arenian-diagnosis]].
