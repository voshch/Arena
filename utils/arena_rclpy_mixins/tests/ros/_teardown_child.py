"""Child process for test_spin_teardown: exercises spin.py teardown paths under a real rclpy context."""

from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback

import rclpy
import rclpy.node
from arena_rclpy_mixins.node import ArenaMixinNode
from arena_rclpy_mixins.spin import create_executor, spin_context, spin_node, start_loop_watchdog
from std_msgs.msg import String
from std_srvs.srv import SetBool

_STALL_S = 12.0
_DEADLINE_BLOCK_S = 5.0
_DEADLINE_S = 1.5
_SLOW_PUBLISH_S = 0.005


class Storm(ArenaMixinNode):
    """Fire-and-forget publish tasks at 100 Hz, the shape of humansim's interpolation loop."""

    async def setup(self) -> None:
        self._pub = self.create_publisher(String, "teardown_probe", 10)
        self._storm = asyncio.create_task(self._run())
        print("READY", flush=True)

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            loop.create_task(self._publish_once())
            await asyncio.sleep(0.01)

    async def _publish_once(self) -> None:
        self._pub.publish(String(data="tick"))


class SlowTeardown(Storm):
    """Storm whose teardown announces itself and takes a second."""

    async def teardown(self) -> None:
        print("TEARDOWN", flush=True)
        await asyncio.sleep(1.0)


class Requester(ArenaMixinNode):
    """Keepalive publish loop, with a sync timer callback that requests shutdown, the shape of the task generator."""

    async def setup(self) -> None:
        self._pub = self.create_publisher(String, "teardown_request", 10)
        self._keepalive = asyncio.create_task(self._run())
        self._timer = self.create_timer(0.3, self._request)

    async def _run(self) -> None:
        try:
            while True:
                await asyncio.sleep(0.005)
                self._pub.publish(String(data="tick"))
        except asyncio.CancelledError:
            pass
        except Exception:
            traceback.print_exc()

    def _request(self) -> None:
        self._timer.cancel()
        self.request_shutdown("test")

    async def teardown(self) -> None:
        print("TEARDOWN", flush=True)


class Stall(ArenaMixinNode):
    """Block the loop past the watchdog threshold, then report back."""

    async def setup(self) -> None:
        print("READY", flush=True)
        self._stall = asyncio.create_task(self._block())

    async def _block(self) -> None:
        time.sleep(_STALL_S)
        print("RESUMED", flush=True)


class Deadline(ArenaMixinNode):
    """Block the loop past a short watchdog deadline; on_deadline exits the process."""

    async def setup(self) -> None:
        loop = asyncio.get_running_loop()
        start_loop_watchdog(loop, self, deadline_s=_DEADLINE_S, on_deadline=lambda: os._exit(7))
        self._stall = asyncio.create_task(self._block())

    async def _block(self) -> None:
        time.sleep(_DEADLINE_BLOCK_S)


class Failing(ArenaMixinNode):
    """An async timer callback raises once the executor spins."""

    async def setup(self) -> None:
        self._timer = self.create_timer(0.05, self._fail)

    async def _fail(self) -> None:
        raise RuntimeError("callback failed")


class Plain(rclpy.node.Node):
    """Plain node with a timer, publishers and a service."""

    def __init__(self, publish_delay_s: float = 0.0) -> None:
        super().__init__("teardown_plain")
        self._publish_delay_s = publish_delay_s
        self._pubs = [self.create_publisher(String, f"teardown_plain_{i}", 10) for i in range(40)]
        self.create_service(SetBool, "teardown_plain_flag", lambda req, res: res)
        self._ready = False
        self.create_timer(0.05, self._tick)

    def _tick(self) -> None:
        for pub in self._pubs:
            pub.publish(String(data="tick"))
            time.sleep(self._publish_delay_s)
        if not self._ready:
            self._ready = True
            print("READY", flush=True)


class Idle(rclpy.node.Node):
    """Node that reports ready from a one-shot timer and then leaves the executor idle."""

    def __init__(self) -> None:
        super().__init__("teardown_idle")
        self._timer = self.create_timer(0.05, self._ready)

    def _ready(self) -> None:
        self._timer.cancel()
        print("READY", flush=True)


class SlowFinalizer:
    """Hold interpreter finalization open after signal handlers are reset."""

    def __del__(self, write=os.write, sleep=time.sleep) -> None:
        write(1, b"FINALIZING\n")
        sleep(1.0)


_FINALIZER: SlowFinalizer | None = None


def main() -> None:
    global _FINALIZER
    mode = sys.argv[1]
    if mode == "async_storm":
        Storm.run_main("teardown_storm")
    elif mode == "async_slow_teardown":
        SlowTeardown.run_main("teardown_slow")
    elif mode == "async_linger":
        Storm.run_main("teardown_storm")
        print("RETURNED", flush=True)
        time.sleep(1.0)
    elif mode == "async_finalize":
        Storm.run_main("teardown_storm")
        _FINALIZER = SlowFinalizer()
    elif mode == "async_request":
        Requester.run_main("teardown_request")
    elif mode == "async_stall":
        Stall.run_main("teardown_stall")
    elif mode == "watchdog_deadline":
        Deadline.run_main("teardown_deadline")
    elif mode == "callback_failure":
        Failing.run_main("teardown_callback_failure")
    elif mode == "sync_events":
        rclpy.init()
        spin_node(Plain(), executor=create_executor())
    elif mode == "sync_events_linger":
        rclpy.init()
        spin_node(Plain(), executor=create_executor())
        print("RETURNED", flush=True)
        time.sleep(1.0)
    elif mode == "sync_global_finalize":
        rclpy.init()
        spin_node(Plain())
        _FINALIZER = SlowFinalizer()
    elif mode == "sync_events_slow":
        rclpy.init()
        spin_node(Plain(_SLOW_PUBLISH_S), executor=create_executor())
    elif mode == "sync_global_slow":
        rclpy.init()
        spin_node(Plain(_SLOW_PUBLISH_S))
    elif mode == "sync_events_idle":
        rclpy.init()
        spin_node(Idle(), executor=create_executor())
    elif mode == "sync_global_idle":
        rclpy.init()
        spin_node(Idle())
    elif mode == "sync_late":
        rclpy.init()
        with spin_context():
            rclpy.shutdown()
            raise RuntimeError("late callback")
    elif mode == "sync_real":
        rclpy.init()
        with spin_context():
            raise RuntimeError("real failure")
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
