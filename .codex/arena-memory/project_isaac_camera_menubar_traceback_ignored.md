---
name: project_isaac_camera_menubar_traceback_ignored
description: "Isaac GUI startup camera_collection.checked AttributeError is NVIDIA's bug, cosmetic, user chose to ignore"
metadata: 
  node_type: memory
  type: project
  originSessionId: af69523e-6daf-4ccb-ae8a-c988ec14047a
---

The one-shot GUI-startup traceback `AttributeError: 'NoneType' object has no attribute 'checked'` in `omni.kit.viewport.menubar.camera-107.0.10` (`camera_menu_container.py:544`, `__camera_changed`) is a missing None-guard in NVIDIA's vendored extension source inside the Isaac image, not Arena code. Exposed because the launcher creates the viewport after app-ready, so the first render-settings notification beats the lazy menubar build; self-heals when the menubar first draws (`_build_camera_collections` re-syncs checked state). User chose to ignore 2026-07-02, don't re-flag. If it ever matters: one-line `if context.camera_collection:` guard via sed in `_meta/docker/features/isaac/Dockerfile`. Related: [[project_isaac_sensor_publish_fix]].
