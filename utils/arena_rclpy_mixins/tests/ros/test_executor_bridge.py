from __future__ import annotations

import asyncio
import threading

import pytest
import rclpy.callback_groups
import rclpy.executors
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from std_msgs.msg import String
from std_srvs.srv import Trigger

EXECUTORS = ("events", "single", "multi")


def _executor(kind: str) -> rclpy.executors.Executor:
    from rclpy.experimental import EventsExecutor

    return {"events": EventsExecutor, "single": rclpy.executors.SingleThreadedExecutor, "multi": rclpy.executors.MultiThreadedExecutor}[kind]()


def _spin(executor: rclpy.executors.Executor) -> threading.Thread:
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return thread


def test_create_executor_is_events_executor_with_entityless_slot():
    from arena_rclpy_mixins.spin import create_executor

    executor = create_executor()
    try:
        assert isinstance(executor, _rclpy.EventsExecutor)
        assert len(executor.get_nodes()) == 1
    finally:
        executor.shutdown()


@pytest.mark.parametrize("kind", EXECUTORS)
def test_async_service_awaits_own_client(kind: str):
    async def main():
        from arena_rclpy_mixins.Async import AsyncNode

        node = AsyncNode(f"bridge_service_{kind}")
        loop_thread = threading.get_ident()
        handler_threads: list[int] = []
        inner = node.create_client_wrapper(Trigger, f"bridge_inner_{kind}", timeout=30.0)

        def _inner(request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
            response.success = True
            response.message = "inner"
            return response

        async def _outer(request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
            handler_threads.append(threading.get_ident())
            result = await inner.call_timeout(Trigger.Request())
            response.success = result is not None and result.success
            response.message = "outer" if response.success else "inner call failed"
            return response

        callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
        node.create_service(Trigger, f"bridge_inner_{kind}", _inner, callback_group=callback_group)
        node.create_service(Trigger, f"bridge_outer_{kind}", _outer, callback_group=callback_group)
        outer = node.create_client_wrapper(Trigger, f"bridge_outer_{kind}", timeout=30.0)
        executor = _executor(kind)
        executor.add_node(node)
        thread = _spin(executor)
        try:
            assert await outer.ensure(timeout_sec=30.0)
            result = await outer.call_timeout(Trigger.Request())
            assert result is not None
            assert result.success, result.message
            assert result.message == "outer"
            assert handler_threads == [loop_thread]
        finally:
            executor.shutdown()
            thread.join(timeout=30.0)
            node.destroy_node()

    asyncio.run(main())


@pytest.mark.parametrize("kind", EXECUTORS)
def test_async_subscription_runs_on_loop_in_order(kind: str):
    async def main():
        from arena_rclpy_mixins.Async import AsyncNode

        node = AsyncNode(f"bridge_subscription_{kind}")
        loop_thread = threading.get_ident()
        received: list[tuple[str, int]] = []
        done = asyncio.Event()
        total = 20

        async def _on_msg(msg: String) -> None:
            received.append((msg.data, threading.get_ident()))
            if len(received) == total:
                done.set()

        topic = f"bridge_topic_{kind}"
        node.create_subscription(String, topic, _on_msg, total)
        pub = node.create_publisher(String, topic, total)
        executor = _executor(kind)
        executor.add_node(node)
        thread = _spin(executor)
        try:
            while pub.get_subscription_count() == 0:
                await asyncio.sleep(0.01)
            for i in range(total):
                pub.publish(String(data=str(i)))
            await asyncio.wait_for(done.wait(), timeout=30.0)
            assert [data for data, _ in received] == [str(i) for i in range(total)]
            assert {ident for _, ident in received} == {loop_thread}
        finally:
            executor.shutdown()
            thread.join(timeout=30.0)
            node.destroy_node()

    asyncio.run(main())


def test_async_service_error_stops_events_executor_spin():
    async def main():
        from arena_rclpy_mixins.Async import AsyncNode
        from arena_rclpy_mixins.spin import create_executor

        node = AsyncNode("bridge_service_error")

        async def _fail(request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
            raise RuntimeError("handler failed")

        node.create_service(Trigger, "bridge_service_error", _fail)
        client = node.create_client_wrapper(Trigger, "bridge_service_error", timeout=1.0)
        executor = create_executor()
        executor.add_node(node)
        errors: list[BaseException] = []

        def _spin_catching() -> None:
            try:
                executor.spin()
            except RuntimeError as e:
                errors.append(e)

        thread = threading.Thread(target=_spin_catching, daemon=True)
        thread.start()
        try:
            assert await client.ensure(timeout_sec=30.0)
            assert await client.call_timeout(Trigger.Request()) is None
            assert [str(e) for e in errors] == ["handler failed"]
        finally:
            executor.shutdown()
            thread.join(timeout=30.0)
            node.destroy_node()

    asyncio.run(main())


def test_async_action_execute_awaits_loop_on_events_executor():
    async def main():
        import rclpy.action
        from arena_rclpy_mixins.Async import AsyncNode
        from arena_rclpy_mixins.spin import create_executor
        from example_interfaces.action import Fibonacci

        node = AsyncNode("bridge_action")
        inner = node.create_client_wrapper(Trigger, "bridge_action_inner", timeout=30.0)

        def _inner(request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
            response.success = True
            return response

        async def _episode(order: int) -> Fibonacci.Result:
            reply = await inner.call_timeout(Trigger.Request())
            return Fibonacci.Result(sequence=[order, int(reply is not None and reply.success)])

        async def _execute(goal_handle: rclpy.action.server.ServerGoalHandle) -> Fibonacci.Result:
            result = await node.loop_future(_episode(goal_handle.request.order))
            goal_handle.succeed()
            return result

        node.create_service(Trigger, "bridge_action_inner", _inner)
        server = rclpy.action.ActionServer(node, Fibonacci, "bridge_action", execute_callback=_execute)
        client = node.create_action_client_wrapper(Fibonacci, "bridge_action", timeout=30.0)
        executor = create_executor()
        executor.add_node(node)
        thread = _spin(executor)
        try:
            assert await client.ensure(timeout_sec=30.0)
            response = await client.send_and_await(Fibonacci.Goal(order=7), timeout_sec=30.0)
            assert response is not None
            assert list(response.result.sequence) == [7, 1]
        finally:
            executor.shutdown()
            thread.join(timeout=30.0)
            server.destroy()
            node.destroy_node()

    asyncio.run(main())
