import asyncio
import contextlib
import datetime
import functools
import math
import typing

import attrs
import builtin_interfaces.msg
import rclpy.callback_groups
import rclpy.clock
import rclpy.node
import rclpy.qos
import rclpy.time
import rclpy.timer
import rosgraph_msgs.msg
from typing_extensions import Self

T = typing.TypeVar('T')

_CLOCK_QOS = rclpy.qos.QoSProfile(
    depth=1,
    reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
    durability=rclpy.qos.DurabilityPolicy.VOLATILE,
    history=rclpy.qos.HistoryPolicy.KEEP_LAST,
)

_NANOSECONDS_PER_SECOND = 10**9


@functools.total_ordering
@attrs.define
class Time:
    """
    Wrapper for builtin_interfaces.msg.Time
    """

    sec: int = attrs.field(converter=int, default=0)
    nanosec: int = attrs.field(converter=int, default=0)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            other = self.parse(other)  # type: ignore
        return self.sec == other.sec and self.nanosec == other.nanosec

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            other = self.parse(other)  # type: ignore
        return (self.sec, self.nanosec) < (other.sec, other.nanosec)

    def __add__(self, other: object) -> Self:
        if not isinstance(other, type(self)):
            other = self.parse(other)  # type: ignore

        return self.from_nanoseconds(self.to_nanoseconds() + other.to_nanoseconds())

    def __sub__(self, other: object) -> Self:
        """Differences may be negative, e.g. across a sim-clock rewind or wall-clock skew."""
        if not isinstance(other, type(self)):
            other = self.parse(other)  # type: ignore
        return self.from_nanoseconds(self.to_nanoseconds() - other.to_nanoseconds())

    # Parsing

    @classmethod
    def from_rclpy(cls, v: rclpy.time.Time) -> Self:
        sec, nanosec = v.seconds_nanoseconds()
        return cls(
            sec=sec,
            nanosec=nanosec,
        )

    @classmethod
    def from_msg(cls, v: builtin_interfaces.msg.Time) -> Self:
        return cls(
            sec=v.sec,
            nanosec=v.nanosec,
        )

    @classmethod
    def from_rosgraph_msg(cls, v: rosgraph_msgs.msg.Clock) -> Self:
        return cls.from_msg(v.clock)

    @classmethod
    def from_float(cls, v: float) -> Self:
        sec = math.floor(v)
        nanosec = int((v - sec) * 1e9)
        return cls(
            sec=sec,
            nanosec=nanosec,
        )

    @classmethod
    def from_nanoseconds(cls, v: int) -> Self:
        """Normalizes so nanosec stays in [0, 1e9), which `to_msg` requires (uint32)."""
        sec, nanosec = divmod(int(v), _NANOSECONDS_PER_SECOND)
        return cls(
            sec=sec,
            nanosec=nanosec,
        )

    @classmethod
    def parse(cls, v: builtin_interfaces.msg.Time | rosgraph_msgs.msg.Clock | rclpy.time.Time | float) -> Self:
        """s.fr
        Create instance from either builtin_interfaces.msg.Time or rclpy.time.Time object.
        """
        if isinstance(v, builtin_interfaces.msg.Time):
            return cls.from_msg(v)
        elif isinstance(v, rosgraph_msgs.msg.Clock):
            return cls.from_rosgraph_msg(v)
        elif isinstance(v, rclpy.time.Time):
            return cls.from_rclpy(v)
        elif isinstance(v, (int, float)):
            return cls.from_float(v)
        else:
            raise TypeError(f'Cannot parse Time from type: {type(v)}')

    # Converting

    def to_rclpy(self) -> rclpy.time.Time:
        """
        Create rclpy.time.Time from self.
        """
        return rclpy.time.Time(
            seconds=self.sec,
            nanoseconds=self.nanosec,
        )

    def to_msg(self) -> builtin_interfaces.msg.Time:
        """
        Create builtin_interfaces.msg.Time from self.
        """
        return builtin_interfaces.msg.Time(
            sec=self.sec,
            nanosec=self.nanosec,
        )

    def to_rosgraph_msg(self) -> rosgraph_msgs.msg.Clock:
        """
        Create rosgraph_msgs.msg.Clock from self.
        """
        return rosgraph_msgs.msg.Clock(clock=self.to_msg())

    def to_seconds(self) -> float:
        """
        Convert to seconds
        """
        return self.sec + self.nanosec / 1e9

    def to_nanoseconds(self) -> int:
        """
        Convert to nanoseconds
        """
        return self.sec * _NANOSECONDS_PER_SECOND + self.nanosec


class TimeNode(rclpy.node.Node):
    """Mixin class to provide clock utilities for rclpy nodes."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.__clock_subscriber = self.create_subscription(
            rosgraph_msgs.msg.Clock,
            '/clock',
            self.__clock_callback,
            _CLOCK_QOS,
        )
        self.__sim_time: Time = Time()
        self.__wall_clock: rclpy.clock.Clock | None = None

    def __clock_callback(self, msg: rosgraph_msgs.msg.Clock):
        """Callback for /clock topic to update node clock when using simulated time.

        Args:
            msg (rosgraph_msgs.msg.Clock): Clock message
        """
        self.__sim_time = Time.from_rosgraph_msg(msg)

    @property
    def sim_time(self) -> Time:
        """Get the latest simulated time received from /clock topic.

        Returns:
            Time: Simulated time
        """
        return Time(self.__sim_time.sec, self.__sim_time.nanosec)

    async def await_forever(
        self,
        awaitable: typing.Awaitable[T],
        what: str,
        warn_period: float = 10.0,
    ) -> T:
        """Unbounded await that narrates the wait on a fixed cadence. The caller owns cancellation."""
        task = asyncio.ensure_future(awaitable)
        loop = asyncio.get_running_loop()
        start = loop.time()
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=warn_period)
                if done:
                    return task.result()
                self.get_logger().warning(f"waiting on {what} ({loop.time() - start:.0f}s)")
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def poll(
        self,
        predicate: typing.Callable[[], bool],
        what: str,
        warn_period: float = 10.0,
        interval: float = 0.1,
    ) -> None:
        """Unbounded predicate poll that narrates the wait on a fixed cadence."""
        loop = asyncio.get_running_loop()
        start = loop.time()
        next_warn = warn_period
        while not predicate():
            await asyncio.sleep(interval)
            elapsed = loop.time() - start
            if elapsed >= next_warn:
                self.get_logger().warning(f"waiting on {what} ({elapsed:.0f}s)")
                next_warn = elapsed + warn_period

    async def arrives(
        self,
        topic: str,
        msg_type: type[T],
        *,
        min_stamp: Time | None = None,
        qos: int | rclpy.qos.QoSProfile = 10,
        warn_period: float = 10.0,
    ) -> T:
        """Temp-subscribe to topic, resolve on first message with header.stamp >= min_stamp
        (any message when min_stamp is None). Unbounded, narrates at warn_period."""
        result: list[T] = []

        def _callback(msg: T) -> None:
            if result:
                return
            if min_stamp is None or Time.from_msg(msg.header.stamp) >= min_stamp:
                result.append(msg)

        sub = self.create_subscription(msg_type, topic, _callback, qos)
        try:
            await self.poll(lambda: bool(result), topic, warn_period=warn_period)
        finally:
            self.destroy_subscription(sub)
        return result[0]

    async def await_sim_step(self, timeout_sec: float | None = None) -> bool:
        """Block until /clock advances past its current value (proof the sim is stepping). False on timeout."""
        start = self.sim_time.to_seconds()

        if timeout_sec is None:
            await self.poll(lambda: self.sim_time.to_seconds() > start, "sim clock step")
            return True

        async def _wait() -> None:
            while self.sim_time.to_seconds() <= start:
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(_wait(), timeout=timeout_sec)
        except TimeoutError:
            return False
        return True

    @property
    def wall_time(self) -> Time:
        """Get the current wall time.

        Returns:
            Time: Wall time
        """
        now = datetime.datetime.now()
        sec = int(now.timestamp())
        nanosec = int(now.microsecond * 1e3)
        return Time(sec=sec, nanosec=nanosec)

    @property
    def time(self) -> Time:
        """Get the current time. Sim time if using ROS time, wall time otherwise.

        Returns:
            Time: Current time
        """
        now = self.get_clock().now()
        sec, nanosec = now.seconds_nanoseconds()
        return Time(sec=sec, nanosec=nanosec)

    @property
    def wall_clock(self) -> rclpy.clock.Clock:
        """Steady wall clock, immune to NTP adjustments. Memoized; reusable across timers."""
        if self.__wall_clock is None:
            self.__wall_clock = rclpy.clock.Clock(clock_type=rclpy.clock.ClockType.STEADY_TIME)
        return self.__wall_clock

    def wall_timer(
        self,
        period_sec: float,
        callback: typing.Callable[[], None],
        *,
        callback_group: rclpy.callback_groups.CallbackGroup | None = None,
    ) -> rclpy.timer.Timer:
        """Create a timer driven by the steady wall clock, regardless of the node's `use_sim_time`."""
        return self.create_timer(
            period_sec,
            callback,
            clock=self.wall_clock,
            callback_group=callback_group,
        )

    @contextlib.contextmanager
    def sim_time_rate(self, rate: float, lifetime: float | None = None) -> typing.Generator[tuple[asyncio.Event, asyncio.Queue[float]], None, None]:
        """Context manager to perform a task at a given simulated time rate.

        Usage:
            ```
            with node.sim_time_rate(10.0) as (done, rate):
                while not done.is_set():
                    dt = await rate.get()
                    # do something
            ```

        Can be canceled by setting the `done` event or exiting the context.

        Args:
            rate (float): Rate in Hz to perform the task.
            lifetime (float | None): Optional lifetime in seconds for the rate context.

        Yields:
            asyncio.Queue: Queue that yields the time delta in seconds at each rate interval.
        """
        interval = 1.0 / rate
        last_time = self.sim_time
        finish_time = last_time + Time.from_float(lifetime) if lifetime is not None else None

        done = asyncio.Event()
        events: asyncio.Queue[float] = asyncio.Queue(maxsize=1000)

        async def _rate_loop():
            nonlocal last_time
            while not done.is_set():
                await asyncio.sleep(0.01)
                now = self.sim_time
                if is_put := (dt := (now - last_time).to_seconds()) >= interval:
                    last_time = now
                    await events.put(dt)
                if finish_time is not None and now >= finish_time:
                    done.set()
                    if not is_put:
                        await events.put(dt)
                    break

        events.put_nowait(0.0)
        loop_task = asyncio.create_task(_rate_loop())
        try:
            yield done, events
        finally:
            loop_task.cancel()
