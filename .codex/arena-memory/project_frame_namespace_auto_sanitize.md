---
name: frame-namespace-auto-sanitize
description: FrameNamespace.auto_sanitize was removed in commit 38f3cb0; Gazebo now sees raw slash-form names like env_0/jackal
metadata: 
  node_type: memory
  type: project
  originSessionId: cddfc3b2-db25-4eb9-8e01-93886dd76f18
---

`FrameNamespace.auto_sanitize()` used to be called at `gazebo_simulator.py` import to rebind `FrameNamespace.__str__` to its `sanitize()` method (slashes -> underscores), so Gazebo entity names appeared as `env_0_jackal`. **That method and its call site were both removed in commit 38f3cb0 ("working robots + manipulators").**

**Current behavior:** Gazebo receives raw realizer output — `env_0/jackal`, slash form, same as Isaac. `GazeboHost.env_prefix` correspondingly should be the default `f"env_{env_id}/"` (the underscore-form override was a leftover from the sanitize era and breaks cleanup).

**Why:** Gazebo Sim does accept `/` in model names; the sanitization was defensive, not strictly required.

**How to apply:** When reasoning about Gazebo entity names, use the slash form. If you see the comment "gazebo does not support slashes" anywhere, it's stale — delete it. `gz model --list` on a live env returns names like `env_0/jackal`, which `startswith("env_0/")` matches but `startswith("env_0_")` does not.
