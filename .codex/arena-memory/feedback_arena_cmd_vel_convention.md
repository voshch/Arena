---
name: Arena cmd_vel uses TwistStamped on ~/cmd_vel
description: When writing or wiring ros2_controllers in Arena, controllers must subscribe to ~/cmd_vel as TwistStamped (not ~/reference); a twist_stamper sits between Nav2 and the controller
type: feedback
originSessionId: 562b37a8-626c-4593-805a-6651517eb139
---
Arena's bringup puts a `twist_stamper` node between Nav2 and every ros2_control controller. The twist_stamper publishes `TwistStamped` to `<controller_name>/cmd_vel` (NOT `~/reference`).

So when writing or configuring any controller in this codebase:

- Topic name is always `~/cmd_vel`, regardless of stamped/unstamped.
- Type is determined by `use_stamped_vel`: `true` -> TwistStamped on `~/cmd_vel`, `false` -> Twist on `~/cmd_vel`.
- For real robots wired through Arena's adapter pipeline, always set `use_stamped_vel: true`.

**Why**: Newer ros2_controllers (mecanum_drive_controller etc.) default to `~/reference` for stamped input. That convention does NOT match Arena. Following it produces a publisher/subscriber type-hash mismatch (twist_stamper publishes TwistStamped on `cmd_vel`, controller subscribes as Twist) and the controller silently never receives motion commands. Burned us once when bringing up arena_swerve_controller on rbvogui (2026-05).

**How to apply**: When writing new ros2_control plugins for Arena, route both stamped and unstamped subscriptions to `~/cmd_vel` (only the type varies). When wiring an existing controller via `control.yaml`, use `use_stamped_vel: true`. Match what husky's diff_drive_controller does, not what newer upstream ros2_controllers default to.
