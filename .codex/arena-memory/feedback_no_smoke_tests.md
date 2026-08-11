---
name: No smoke tests or runtime checks
description: Don't run build/launch/smoke tests or any runtime commands when verifying changes in this project — static analysis only
type: feedback
originSessionId: 2b2c6597-2983-460e-8fa6-e1ab40245030
---
Don't run smoke tests, runtime launches, `colcon build`, `python -c "import ..."` sanity checks, or any command that actually executes task_generator code. Verification is static analysis only (grep, file reads, import-section review).

**Why:** User runs the runtime side themselves and explicitly rejected a proposed import-sanity check during rounds 1–3 verification with "no checks. just static analysis". They want the static analysis layer cleanly separated from runtime validation, which they handle manually.

**How to apply:** When writing refactor plans or verification steps, omit smoke-test / build / runtime-launch items entirely. Do not propose them in plan.md sections. If runtime verification feels necessary, describe *what the user should check when they run it themselves*, but do not offer to run it.
