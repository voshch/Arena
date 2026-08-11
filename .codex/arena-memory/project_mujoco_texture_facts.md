---
name: project_mujoco_texture_facts
description: "Source-verified MuJoCo 3.9.0 renderer facts behind arena_mujoco texture wiring (texrepeat 2x factor, geom groups, plane ghost, rgba modulate)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2778f08c-e772-48f3-aeb1-5fd859ba310f
---

Verified against MuJoCo 3.9.0 source (render_gl3.c, xml_urdf.cc, user_objects.cc) during the 2026-06-10 texture implementation, the XML docs are misleading on several of these:

- **texrepeat with texuniform=true tiles every 2/texrepeat world meters**, not 1/texrepeat (object space is [-1,1] over half-extents). `TILE_M` in mesh_mat.py encodes this as `texrepeat = 2/TILE_M`.
- **MuJoCo's URDF importer puts visual geoms in group 1** (xml_urdf.cc:542), collision geoms in group 0. Groups 0-2 are default-visible. Arena ceilings therefore use group 2 (`CEILING_GROUP` in SpawnCeilings.py), hidden only in the env-0 viewer's `opt.geomgroup`, robot cameras keep them.
- **Plane geoms render opaque from +Z and as a 30%-alpha ghost from behind** (render_gl3.c rgba[3]*=0.3 when camera isBehind). This is the one-way ceiling mechanism, MuJoCo can do per-camera ceiling hiding unlike Isaac ([[project_isaac_no_oneway_ceilings]]).
- **Non-default geom rgba does NOT disable texturing** (no matid=-1 folklore in 3.9.0), it multiplies the texture via GL_MODULATE. Sentinel to defer color to the material: rgba (0.5, 0.5, 0.5, 1).
- **No per-geom shadow flag**, shadow pass renders all faces culling-disabled, so Ceiling.cast_shadows stays unhonorable.
- **OBJ decoder**: reads vt (V-flips itself), ignores .mtl entirely, decodes only shape[0]. So MTL is parsed Arena-side (`obj_albedo`) and trimesh conversions must export a single Trimesh, never a Scene.
- **Data textures**: PIL `tobytes()` top-row-first RGB matches exactly, set colorspace=SRGB explicitly (data path skips PNG sRGB-chunk detection). File loading is PNG/KTX only, JPG must go via data.
- Pins live in _meta/features/mujoco: mujoco==3.9.0 (spec.assets dict removed after 3.9), trimesh>=4,<5.

**Status**: textures implemented for walls/floors/ceilings (MDL albedo, tiled), OBJ obstacles (dormant until OBJ DB rebuild), robot meshes (DAE->OBJ via trimesh). Albedo only, native renderer has no PBR. Unverified at runtime: nothing rendered yet as of 2026-06-10.
