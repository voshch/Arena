---
name: project_isaac_no_oneway_ceilings
description: "Isaac/RTX can't do one-way ceilings; they spawn invisible, Gazebo does the real one-way"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a889786-059c-4191-a71d-fcf0da3f6ff9
---

One-way ceilings (opaque from below, transparent from above) work in Gazebo but NOT Isaac.

- **Gazebo (ogre2):** free via a `<plane>` primitive with `<normal>0 0 -1</normal>` + `<cast_shadows>false</cast_shadows>`. Rasterizer back-face-culls. Works.
- **Isaac (RTX 5.1):** dead end. Confirmed via RTX docs + the installed Omniverse:
  - RTX ignores USD `doubleSided`; back-face culling only via global `/rtx/hydra/faceCulling/enabled`, which gets reset on engine init AND still renders two-sided even when set True with `doubleSided=False` (likely material/path-tracer override).
  - No per-camera / per-render-product visibility: cannot show a prim to robot cameras but hide it from the viewport. USD `visibility` is global, `purpose` only gives the reverse split (proxy = viewport-yes/sensor-no).
- **Resolution:** `arena_isaac/.../services/SpawnCeilings.py` spawns Isaac ceilings **invisible** (`MakeInvisible()`), prim left in the stage to toggle for inspection. Don't re-attempt culling or per-camera tricks, they were all exhausted.

Don't burn time re-investigating this in Isaac. See [[feedback_no_smoke_tests]].
