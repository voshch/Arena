---
name: project_isaac_driver_595_crash
description: Isaac Sim 5.1 crashes at RTX startup on NVIDIA driver 595/610; pin host driver 580
metadata: 
  node_type: memory
  type: project
  originSessionId: 8da6e623-9475-4290-b675-35ed6b342336
---

Isaac Sim 5.1 crashes during renderer init (fatal in `librtx.scenedb.plugin.so` at `carbOnPluginStartup`, before any Arena scene/world/robot loads) on NVIDIA driver **595.x and 610.x**. Reproduced on this box (Ubuntu 26.04 resolute, kernel 7.0, RTX 2080 Ti). Headless does not help (RTX backend loads either way), and a container rebuild does not help (caches live in the container writable layer and are already fresh; the bug is host-side).

Confirmed upstream: github.com/isaac-sim/IsaacSim/issues/537 (595 crashes, 580 works). It is NOT an Arena bug and NOT a stale-cache issue.

**Fix:** pin host driver to the **580** branch. On 26.04 it is a real apt package: `nvidia-driver-580` (580.159.03), built for kernel 7.0. `apt purge '^nvidia-.*'` → `apt install nvidia-driver-580` → `apt-mark hold nvidia-driver-580 nvidia-dkms-580` → reboot. NOTE on 26.04 `nvidia-driver-590` is a transitional alias for 595, so "downgrade to 590" via apt gives you 595 (still broken). Verified working on 580.

When debugging Isaac startup crashes: pull the py-spy trace + crash report out of the (possibly exited) container with `docker cp arena-arena_ws-isaac-1:'/isaac-sim/kit/data/Kit/Isaac-Sim Python/5.1/.' <dest>` (the `.py.txt` shows the Python stack at crash time).
