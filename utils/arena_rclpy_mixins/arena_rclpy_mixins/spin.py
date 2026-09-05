"""Process-spin helpers: clean rclpy teardown across SIGINT/SIGTERM."""

from __future__ import annotations

import asyncio
import asyncio.base_subprocess
import contextlib
import errno
import faulthandler
import gc
import signal
import threading
import time
import traceback
import typing
from collections.abc import Callable, Iterator

import rclpy
import rclpy.executors
import rclpy.node
from rclpy.exceptions import InvalidHandle
from rclpy.executors import ExternalShutdownException
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.signals import SignalHandlerOptions

if typing.TYPE_CHECKING:
    from arena_rclpy_mixins import ArenaMixinNode

_LOOP_STALL_S = 10.0
_LOOP_BEAT_S = 1.0


@contextlib.contextmanager
def spin_context(
    *,
    executor: rclpy.executors.Executor | None = None,
) -> Iterator[None]:
    """Wrap a custom mainloop: swallow shutdown-time exceptions, shut down executor + rclpy on exit."""
    try:
        yield
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        with contextlib.suppress(KeyboardInterrupt):
            if executor is not None:
                executor.shutdown()
            if rclpy.ok():
                rclpy.shutdown()


def spin_node(node: rclpy.node.Node, **kwargs: object) -> None:
    """Spin a single node until shutdown."""
    executor = kwargs.get("executor")
    with spin_context(**kwargs):
        try:
            rclpy.spin(node, executor=executor)
        finally:
            with contextlib.suppress(KeyboardInterrupt):
                node.destroy_node()


def _close_subprocess_transports() -> None:
    # gc walk: pipe transports are in loop._transports but BaseSubprocessTransport itself isn't.
    for obj in gc.get_objects():
        if isinstance(obj, asyncio.base_subprocess.BaseSubprocessTransport):
            with contextlib.suppress(Exception):
                obj.close()


def _suppress_shutdown_noise(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    exc = context.get("exception")
    if isinstance(exc, asyncio.InvalidStateError):
        return
    if not rclpy.ok():
        if isinstance(exc, AssertionError | InvalidHandle | _rclpy.RCLError):
            return
        if isinstance(exc, OSError) and exc.errno == errno.EBADF:
            return
    loop.default_exception_handler(context)


def _watch_loop(
    loop: asyncio.AbstractEventLoop,
    node: ArenaMixinNode,
    stop: threading.Event,
    *,
    deadline_s: float | None = None,
    on_deadline: Callable[[], None] | None = None,
) -> None:
    """Dump all threads once when the loop stops beating for _LOOP_STALL_S; past deadline_s, dump again and call on_deadline once."""
    last_beat = time.monotonic()
    dumped = False
    deadline_fired = False

    def _beat() -> None:
        nonlocal last_beat
        last_beat = time.monotonic()

    while not stop.wait(_LOOP_BEAT_S):
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(_beat)
        stalled = time.monotonic() - last_beat
        if stalled < _LOOP_STALL_S:
            dumped = False
        elif not dumped:
            dumped = True
            faulthandler.dump_traceback(all_threads=True)
            if not stop.is_set():
                node.get_logger().warning(f"event loop stalled for {stalled:.0f}s, dumping threads")
        if deadline_s is not None and not deadline_fired and stalled >= deadline_s:
            deadline_fired = True
            faulthandler.dump_traceback(all_threads=True)
            with contextlib.suppress(Exception):
                node.get_logger().warning(f"event loop stalled for {stalled:.0f}s past deadline {deadline_s:.0f}s")
            if on_deadline is not None:
                on_deadline()


def start_loop_watchdog(
    loop: asyncio.AbstractEventLoop,
    node: ArenaMixinNode,
    *,
    deadline_s: float | None = None,
    on_deadline: Callable[[], None] | None = None,
) -> threading.Event:
    """Start the loop-stall watchdog thread; returns its stop event."""
    stop = threading.Event()
    threading.Thread(
        target=_watch_loop,
        args=(loop, node, stop),
        kwargs={"deadline_s": deadline_s, "on_deadline": on_deadline},
        name="loop_watchdog",
        daemon=True,
    ).start()
    return stop


async def async_main(
    *,
    node_factory: Callable[[], ArenaMixinNode],
    aiomonitor: bool = False,
) -> None:
    """Standard rclpy+asyncio entry: build node, spin in worker, await forever, tear down."""
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_suppress_shutdown_noise)

    executor = rclpy.executors.MultiThreadedExecutor()
    node = node_factory()
    node.event_loop = loop
    executor.add_node(node)

    watchdog_stop = start_loop_watchdog(loop, node)

    def _spin() -> None:
        while rclpy.ok():
            try:
                executor.spin()
                return
            except ExternalShutdownException:
                return
            except InvalidHandle:
                continue
            except _rclpy.RCLError:
                if rclpy.ok():
                    raise
                return

    spin_future = loop.run_in_executor(None, _spin)

    async def _app() -> None:
        await node.setup()
        await asyncio.Event().wait()

    app_task = asyncio.create_task(_app())

    teardown_started = asyncio.Event()

    async def _graceful_shutdown(sig_name: str) -> None:
        if teardown_started.is_set():
            return
        teardown_started.set()
        node.get_logger().info(f"received {sig_name}, tearing down")
        try:
            await asyncio.wait_for(node.teardown(), timeout=5.0)
        except Exception as e:
            node.get_logger().warning(f"teardown raised: {e!r}")
        app_task.cancel()

    def _c_sig_handler(signum: int, _frame: object) -> None:
        name = signal.Signals(signum).name
        loop.call_soon_threadsafe(lambda: loop.create_task(_graceful_shutdown(name)))

    for sig, name in ((signal.SIGINT, "SIGINT"), (signal.SIGTERM, "SIGTERM")):
        loop.add_signal_handler(
            sig,
            lambda n=name: loop.create_task(_graceful_shutdown(n)),
        )
        signal.signal(sig, _c_sig_handler)
        signal.siginterrupt(sig, False)

    cm: contextlib.AbstractContextManager[object] = contextlib.nullcontext()
    if aiomonitor:
        config = node.aiomonitor_config()
        if config is not None:
            import aiomonitor as _aiomonitor

            try:
                cm = _aiomonitor.start_monitor(loop=loop, locals=locals(), **config)
            except OSError as e:
                node.get_logger().warning(f"aiomonitor disabled: {e!r}")

    try:
        with cm:
            done, _ = await asyncio.wait([spin_future, app_task], return_when=asyncio.FIRST_COMPLETED)

        if spin_future in done:
            spin_future.result()
        if app_task in done:
            app_task.result()
    except asyncio.CancelledError:
        node.get_logger().info("Shutting down.")
    except Exception:
        node.get_logger().error(traceback.format_exc())
        raise
    finally:
        with contextlib.suppress(Exception):
            await node.kill_launches()

        current = asyncio.current_task()
        pending = [t for t in asyncio.all_tasks(loop) if t is not current and not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=2.0)

        executor.shutdown()
        with contextlib.suppress(Exception):
            await spin_future

        watchdog_stop.set()
        executor.remove_node(node)
        node.destroy_node()
        _close_subprocess_transports()
        gc.collect()
        rclpy.try_shutdown()


def run_main(
    node_cls: type[ArenaMixinNode],
    /,
    *args: object,
    aiomonitor: bool = False,
    **kwargs: object,
) -> None:
    """Sync entry. Use as `def main(): run_main(MyNode)` or via the classmethod."""
    try:
        asyncio.run(
            async_main(
                node_factory=lambda: node_cls(*args, **kwargs),
                aiomonitor=aiomonitor,
            )
        )
    except KeyboardInterrupt:
        pass
