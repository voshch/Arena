---
name: Check arena_rclpy_mixins before hand-rolling
description: Before writing any ROS-adjacent glue (namespaces, async bridges, services, lifecycle, params, time), scan arena_rclpy_mixins for an existing utility
type: feedback
originSessionId: 481f48ce-0d46-4cc0-b076-1c025647b313
---
Before hand-rolling ROS glue, read `utils/arena_rclpy_mixins/arena_rclpy_mixins/`. Using stdlib-ish patterns or f-string ad-hoc code when a project utility already exists is bloat the user will call out.

**Why:** the user maintains this module as the shared idiom layer; duplicating its functionality (e.g. `f"/{ns}/foo"` when `Namespace(ns)("foo")` exists, `asyncio.wrap_future` when `AsyncNode.await_ros` exists) creates inconsistency and bugs (double-slash action names, `AssertionError: concurrent.futures.Future is expected`).

**How to apply — scan these modules first:**

- `shared.py` — `Namespace(str)` subclass with `__call__(*parts)` for join + double-slash cleanup; `FrameNamespace` (sanitize for tf); `ParamNamespace` (dotted for parameters). Use this everywhere you construct topic/node/namespace paths. Never write `f"/{ns}/..."`.
- `Async.py` — `AsyncNode` (await_ros, wait_for_service_async, wait_for, sync_wrap, syncify, do_launch, create_client_wrapper, create_action_client_wrapper); `ClientWrapper[ServiceT]` (call_timeout, ensure); `ActionClientWrapper[ActionT]` (send_goal, await_result, send_and_await, cancel, ensure); `AsyncUtil.timeout`; `AsyncLaunchManager`.
- `LifecycleClient.py` — lifecycle state queries and waits (`get_lifecycle_state_async`, `wait_for_lifecycle_state_async`).
- `ServiceNamespace.py` — namespace helpers tailored for service paths.
- `Time.py` — `TimeNode` mixin for sim-time-aware nodes.
- `ROSParamServer.py` — ROS parameter declaration/access helpers.

**Before writing new utility code in arena_robots / task_generator / etc., first check if arena_rclpy_mixins already has it. If it's missing and generic enough, add it to arena_rclpy_mixins (with the project's style — see ClientWrapper as the template) rather than inlining in the consumer package.**
