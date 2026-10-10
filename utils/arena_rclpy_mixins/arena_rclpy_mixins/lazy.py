"""Publishers and subscriptions that skip the work of a message nobody subscribes to."""

from collections.abc import Callable, Sequence

import rclpy.logging
import rclpy.node
import rclpy.publisher
import rclpy.subscription
from rclpy.clock import Clock, ClockType
from rclpy.qos import QoSDurabilityPolicy, QoSProfile


class LazyPublisher[T]:
    """Publisher that builds and sends a message only while the topic has a subscriber. A latched publisher always sends."""

    def __init__(self, publisher: rclpy.publisher.Publisher) -> None:
        self._publisher = publisher
        self._eager = publisher.qos_profile.durability != QoSDurabilityPolicy.VOLATILE
        if self._eager:
            rclpy.logging.get_logger("lazy_publisher").warning(f"{publisher.topic_name} is latched and cannot be lazy, publishing every message")

    @property
    def publisher(self) -> rclpy.publisher.Publisher:
        return self._publisher

    @property
    def wanted(self) -> bool:
        return self._eager or self._publisher.get_subscription_count() > 0

    def publish(self, build: Callable[[], T]) -> bool:
        """Publish build() if wanted, returning whether it was sent."""
        if not self.wanted:
            return False
        self._publisher.publish(build())
        return True


class LazySubscription[T]:
    """Subscription that exists only while a publisher it feeds is wanted, checked every period_s of wall time."""

    def __init__(self, node: rclpy.node.Node, feeds: LazyPublisher | Sequence[LazyPublisher], msg_type: type[T], topic: str, callback: Callable[[T], None], qos_profile: QoSProfile | int, period_s: float = 1.0) -> None:
        self._node = node
        self._feeds = (feeds,) if isinstance(feeds, LazyPublisher) else tuple(feeds)
        self._args = (msg_type, topic, callback, qos_profile)
        self._subscription: rclpy.subscription.Subscription | None = None
        self._timer = node.create_timer(period_s, self._sync, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self._sync()

    @property
    def active(self) -> bool:
        return self._subscription is not None

    def _sync(self) -> None:
        wanted = any(feed.wanted for feed in self._feeds)
        if wanted and self._subscription is None:
            self._subscription = self._node.create_subscription(*self._args)
        elif not wanted and self._subscription is not None:
            self._node.destroy_subscription(self._subscription)
            self._subscription = None
