---
name: feedback-cap-specific-on-cap-base
description: "Cap-specific reset/teleport/joint logic belongs on the cap base class (MobileAdapter, future ArmAdapter), not on RobotManager"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0d328bf5-f691-4d7f-b590-4bb1e11cc40e
---

Cap-specific behavior (mobile teleport, arm joint reset, etc.) belongs on a cap-base adapter class, not on the cap-agnostic `RobotManager`. `RobotManager.move` stays as a generic utility, but `RobotManager.reset` must not call it: it must fan out `on_reset` to adapters, and each cap base (`MobileAdapter`) supplies the shared default.

**Why:** Not every robot is mobile. Baking `await self.move(ctx.start_pose)` into `RobotManager.reset` assumes a body pose to teleport, which is a mobile-cap concern; arm-only / drone-only robots would either crash or silently do the wrong thing. The user pushed back hard on this exact framing.

**How to apply:** When adding cap-specific episode-boundary logic, hang it off the cap base class (`MobileAdapter`, `ArmAdapter`). Concrete adapter overrides chain `super().on_reset(...)` before adding stack-specific work. The `super()` discipline is just discipline, same as everywhere else. Do *not* try to bypass this by moving cap logic up into `RobotManager` just because it's "harder to forget" there.

Related: [[feedback-no-justifying-comments]], [[feedback-match-comment-density]].
