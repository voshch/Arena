---
name: host-python-314-zombie-venv
description: "Host was dist-upgraded to python 3.14 only, no ROS on host, .venv base interpreter is broken (cp312 packages under 3.14 base)"
metadata: 
  node_type: memory
  type: project
  originSessionId: edfe0404-d61a-4251-94a5-f6104c1f85f1
  modified: 2026-08-04T15:27:17.425Z
---

As of 2026-08-04 the host has only /usr/bin/python3.14 (3.12 removed by a dist upgrade) and no /opt/ros at all - ROS exists only in the container. Consequences: the shared `.venv`'s pyvenv.cfg says 3.12.3 but its `bin/python3` now resolves to /usr/bin/python3.14, so the venv is a zombie on the host (cp312 site-packages under a 3.14 base). The `.installed` marker cannot detect this class of breakage (marker present, interpreter broken). Venv refresh is explicit by design: `arena update` syncs, `arena repair` rebuilds (container always, host only greenfield), source NEVER provisions, it prints a one-line not-ready status (stamp-based staleness detection and source-time provisioning were both rejected 2026-08-04). Fresh containers provision once via the entrypoint first-boot (flock-serialized, `_arena_venv_provision`), interactive entries skip past it. A venv-seed-in-image scheme was built on a wrong broken-network theory and reverted: the real fresh-install hang was N concurrent first-boots livelocking on uv's own locks. Open debt item, container-side venv is unaffected.

**Why:** explains host-side python weirdness and why ROS verbs can never run on the host.

**How to apply:** don't diagnose host import failures as regressions, and validate anything ROS-related in the container ([[shared-venv-provisioning]]).
