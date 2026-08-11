---
name: arena_rclpy_mixins is the async bridge
description: Use arena_rclpy_mixins idioms (AsyncNode.await_ros, ClientWrapper, LifecycleClient) for bridging rclpy futures to asyncio, not stdlib asyncio.wrap_future
type: feedback
originSessionId: 481f48ce-0d46-4cc0-b076-1c025647b313
---
When code in this workspace needs to bridge `rclpy.task.Future` into an asyncio context, use the project's own helpers in `utils/arena_rclpy_mixins/arena_rclpy_mixins/Async.py`, not `asyncio.wrap_future`.

**Why:** `rclpy.task.Future` is not a `concurrent.futures.Future`, so `asyncio.wrap_future` trips its isinstance assert (`AssertionError: concurrent.futures.Future is expected, got <rclpy.task.Future>`). The project already has the correct bridge — `AsyncNode.await_ros(ros_future)` — which uses `add_done_callback` + `loop.call_soon_threadsafe` to signal an asyncio future.

**How to apply:**
- **For the entry point of any program:** subclass **`ArenaMixinNode`** (the megaclass that composes `ROSParamServer + LifecycleClient + AsyncLifecycleClient + ServiceNamespace + TimeNode + rclpy.node.Node`) and use `MyNode.run_main(*args, **kwargs)`. The `run_main` classmethod lives on `ArenaMixinNode`, **not** on `AsyncNode`. `AsyncNode` is just `TimeNode + Node` for ad-hoc in-process spinning; subclasses that want the standard process-lifetime entry need `ArenaMixinNode`. The classmethod owns `rclpy.init`, the MultiThreadedExecutor, the spin worker, signal handlers, `node.setup()` / `node.teardown()` lifecycle, pending-task cancellation, and shutdown. **Never hand-roll** `MultiThreadedExecutor() + threading.Thread(target=spin_executor) + asyncio.run(...)` — that path drifts from the official lifecycle, mishandles SIGINT/SIGTERM, mis-types `executor.shutdown(wait=False)` (it's `timeout_sec`), and leaks tasks. To exit early from a one-shot setup: `raise SystemExit(code)` from inside `setup()`, then catch it in your `cli_main` to map to the process exit code.
- For actions: `AsyncNode.create_action_client_wrapper(action_type, name)` returns an `ActionClientWrapper` with `.send_goal()` / `.send_goal_timeout()` / `.await_result()` / `.send_and_await()` / `.cancel()` / `.ensure()`. Don't hand-roll `send_goal_async` + `await_ros` + `get_result_async`.
- For services: `AsyncNode.create_client_wrapper(srv_type, name)` returns a `ClientWrapper` with `.call_timeout()` / `.ensure()` / `.call_timeout_sync()` — don't hand-roll.
- For a raw rclpy future from a custom callback: `await self._node.await_ros(ros_future)` (where `_node` is an `AsyncNode`). Bridge of last resort.
- For lifecycle state queries/waits: `AsyncNode.get_lifecycle_state_async` / `wait_for_lifecycle_state_async` (see LifecycleClient module).
- Blocking-from-sync: `AsyncNode.wait_for(awaitable)` or `AsyncNode.sync_wrap(fn)` / `syncify(fn)`.
- Async timeouts: `AsyncUtil.timeout(coro, timeout_sec)`.
- Launching a `LaunchDescription` from async code: `AsyncNode.do_launch(ld)` (uses `AsyncLaunchManager` internally).
