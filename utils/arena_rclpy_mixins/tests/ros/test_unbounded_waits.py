from __future__ import annotations

import asyncio

import pytest


def _make_node(name: str):
    from arena_rclpy_mixins.Async import AsyncNode

    return AsyncNode(name)


def test_await_forever_returns_result():
    async def main():
        node = _make_node("loud_await_result")
        try:

            async def value():
                await asyncio.sleep(0.05)
                return 42

            return await node.await_forever(value(), "test value", warn_period=0.01)
        finally:
            node.destroy_node()

    assert asyncio.run(main()) == 42


def test_await_forever_propagates_exception():
    async def main():
        node = _make_node("loud_await_raise")
        try:

            async def boom():
                await asyncio.sleep(0.01)
                raise RuntimeError("boom")

            with pytest.raises(RuntimeError, match="boom"):
                await node.await_forever(boom(), "test failure", warn_period=0.01)
        finally:
            node.destroy_node()

    asyncio.run(main())


def test_await_forever_cancel_cancels_inner():
    async def main():
        node = _make_node("loud_await_cancel")
        try:
            started = asyncio.Event()
            cancelled = asyncio.Event()

            async def forever():
                started.set()
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            waiter = asyncio.ensure_future(node.await_forever(forever(), "test forever", warn_period=0.01))
            await started.wait()
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            await asyncio.wait_for(cancelled.wait(), timeout=1.0)
        finally:
            node.destroy_node()

    asyncio.run(main())


def test_poll_returns_when_predicate_true():
    async def main():
        node = _make_node("loud_poll")
        try:
            flag = {"ready": False}

            async def flip():
                await asyncio.sleep(0.05)
                flag["ready"] = True

            flipper = asyncio.ensure_future(flip())
            await asyncio.wait_for(
                node.poll(lambda: flag["ready"], "test flag", warn_period=0.01, interval=0.005),
                timeout=1.0,
            )
            await flipper
        finally:
            node.destroy_node()

    asyncio.run(main())
