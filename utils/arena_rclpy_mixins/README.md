# arena_rclpy_mixins

Shared ROS 2 / asyncio glue for the Arena-Rosnav workspace. Provides the
`ArenaMixinNode` megaclass, typed ROS parameter helpers, lifecycle client
utilities, namespace tools, and spin/teardown helpers. The CLAUDE.md rule:
**scan this package before writing new ROS glue** (namespaces, async, lifecycle,
params, time). If a facility already exists here, use it instead of
hand-rolling.

## Public API

Sourced from `__init__.py` plus the submodules it re-exports.

| Name | Kind | What it does | When to use it |
| --- | --- | --- | --- |
| `ArenaMixinNode` | class | All mixins composed with `rclpy.node.Node`; override `setup()` / `teardown()` | Base class for every Arena node |
| `ROSParamServer` | mixin | Callback-on-set parameter server, the live handle behind `declare` | Nodes whose parameters change at runtime |
| `params` | module | `Param[T]`, `ParamGroup`, `declare`, the constraints, `declare_launch_arguments` | Every node parameter and the launch arguments derived from them |
| `LifecycleClient` | mixin | Sync `get_lifecycle_state`, `change_lifecycle_state`, `wait_for_lifecycle_state` | Driving a peer lifecycle node from sync code |
| `AsyncLifecycleClient` | mixin | Async `*_async` variants of the above via `ClientWrapper` | Driving a peer lifecycle node from `async` context |
| `ServiceNamespace` | mixin | `service_namespace(*parts)`: node-FQN-prefixed `Namespace` | Correct service/topic prefixing (rclpy ignores node namespace for services) |
| `TimeNode` | mixin | `.sim_time`, `.wall_time`, `.time`, `.wall_clock`, `.wall_timer()`, `.sim_time_rate()` | Any node that needs clock utilities |
| `Time` | class | Arithmetic, comparable, converts between `rclpy.time.Time`, `builtin_interfaces.msg.Time`, `rosgraph_msgs.msg.Clock`, and `float` | Time math without rclpy churn |
| `AsyncNode` | class | `await_ros()`, `loop_future()`, `wait_for()`, `sync_wrap()`, `ros_callback()`, `create_client_wrapper()`, `create_action_client_wrapper()`, `do_launch()` | Nodes mixing async and sync code |
| `ClientWrapper` | class | Async `call_timeout()` / sync `call_timeout_sync()` around a service client | Service calls with timeout logging |
| `ActionClientWrapper` | class | `send_goal()`, `send_goal_timeout()`, `await_result()`, `send_and_await()`, `cancel()`, `ensure()` | Action client with timeout wrappers |
| `AsyncUtil` | class | `AsyncUtil.timeout(coro, sec)`: `asyncio.wait_for` that returns `None` on timeout | One-off timeout wrapper |
| `AsyncLaunchManager` / `LaunchHandle` | class | `launch_description()` returns a `LaunchHandle` (awaitable to completion; `shutdown()` gracefully terminates child processes); `kill_all()` shuts down all. Environment actions act on a per-launch copy of `os.environ` (`IsolatedEnvironmentContext`), never on the process environment | Nodes that programmatically launch sub-processes |
| `LazyPublisher` | class | Wraps a volatile publisher: `.wanted` and `.publish(build)` build and send only while the topic has a subscriber. A latched publisher is wrapped with a warning and always sends | Viewer and debug topics whose messages are costly to build. Never lockstep channels, heartbeats or control topics |
| `LazySubscription` | class | Subscribes only while the `LazyPublisher` it feeds is wanted, checked once per second of wall time, so the upstream publisher sees no subscriber either | Relays and converters whose only output is a lazy topic |
| `Namespace` | class | `str` subclass with `/`-join `__call__`, `.simulation_ns`, `.robot_ns`, `.remove_double_slash()` | Building topic / service paths |
| `FrameNamespace` | class | `Namespace` subclass with `.sanitize()` (replaces non-alphanumeric with `_`) | TF frame name construction |
| `ParamNamespace` | class | `Namespace` with `.`-join `__call__`, converts to/from slash namespaces | ROS param key construction |
| `ClassRegistry` | class | Lazy `key -> class` registry with `.register(key)` decorator, cached on first `.get(key)` | Kind-based class lookup (e.g. simulator kind to class) |
| `FactoryRegistry` | class | Like `ClassRegistry` but factory is called fresh on every `.get(key, *args)` | Per-call factory dispatch |
| `AsyncFactoryRegistry` | class | Async version of `FactoryRegistry`; `.get()` returns an `Awaitable` | Async factory dispatch |

## Mixins and `ArenaMixinNode`

`ArenaMixinNode` inherits `ROSParamServer`, `LifecycleClient`,
`AsyncLifecycleClient`, `ServiceNamespace`, `TimeNode`, and
`rclpy.node.Node` in MRO order. Use it as the sole base for Arena nodes:

```python
class MyNode(ArenaMixinNode, rclpy.lifecycle.LifecycleNode):
    async def setup(self) -> None: ...   # called once by run_main
    async def teardown(self) -> None: ...  # called on SIGINT/SIGTERM or request_shutdown()
```

`run_main` (classmethod) is the standard entry point: `def main(): MyNode.run_main()`.

## Async bridge

`AsyncNode.await_ros(ros_future)` bridges a rclpy `Future` into the asyncio
event loop without blocking. Use it (and `ClientWrapper` / `ActionClientWrapper`
built on top of it) instead of `asyncio.wrap_future` or hand-rolled
`run_coroutine_threadsafe` chains (the rclpy bridge is subtle and the wrappers
handle thread-safety and timeout logging correctly).

`create_subscription` and `create_service` on `AsyncNode` transparently accept
`async def` callbacks via `ros_callback()`, so callers never need to bridge those
manually. The callback runs on the event loop while the executor holds only an
rclpy task awaiting `loop_future()`, so it may await ROS futures (service
calls, `/clock`) even on a single-threaded executor. Under `EventsExecutor`,
callback groups do not serialize these coroutines: a slow handler overlaps the
next message's.

`wait_for(future)` submits a coroutine to the node's event loop and blocks the
calling thread until done. It warns if called from the event loop thread itself
(would deadlock). Called from an executor callback it holds that executor
thread, so on a single-threaded executor the awaited coroutine must not need
ROS callbacks.

## Time

`TimeNode` subscribes to `/clock` and exposes `.sim_time` (latest simulated
time), `.wall_time` (datetime-based wall clock), and `.time` (delegates to the
node clock, respecting `use_sim_time`). `wall_timer(period, cb)` creates a
timer driven by the steady wall clock regardless of `use_sim_time`. The `Time`
value type supports arithmetic, comparison, and round-trips to all ROS time
representations.

## Parameters

`params.py` is the one way a node parameter is declared. A `Param[T](name, default,
description, constraint=..., parse=...)` carries its ROS type (read from the default,
or from `type_` for an empty list or a required parameter with default `None`), its
description, and one constraint: `Range(lo, hi, step, lo_open, hi_open)` on numbers
(`None` = unbounded), `Choices(*values, many=False)` on strings (an Enum default
derives the choices and the parse), `Catalog(name)` for values a live catalog lists,
`Pair(lo, hi)` for a `[min, max]` array (one element reads as both, two read sorted
through the `pair` parse), and `Text(sketch=False)` for multiline text. Float
parameters reject NaN and infinity unless the default or a bound is infinite.
`parse` turns the ROS value into `T` on the first read after a change, never at
declaration.

`declare(node, param)` works on any `rclpy.node.Node`: it declares with the
descriptor (a launch override keeps its value and gets the descriptor, an integer
override of a double is redeclared as float), checks the current value against the
constraint and raises `RuntimeError` naming the parameter when it does not fit,
registers the validator for later sets, and returns the live handle (`.name`,
`.value`, `.destroy`). The same name declared again on the same node returns the
same handle. On a `ROSParamServer` the handle is the cached, callback-driven
`ROSParam`.

`ParamGroup` declares `Param[T]` class attributes on a node at construction,
`ParamGroup(node, namespace="task.random", role="array")`, every name prefixed by
the namespace, one instance per node, namespace and role. `PerRole` gives a
parameter a default per role name. `params()`, `by_name(namespace)` and
`defaults()` read the declarations without a node, `values()` the live values,
`configure(config, *groups, **given)` builds an attrs config from either.
`declare_launch_arguments(group.by_name(), prefix=..., reserved=...)` turns every
parameter not in `reserved` into an empty-default launch argument with the
description and, for `Choices`, the choices. `parse_constraint(descriptor)` and
`constraint_json(constraint)` give consumers (`episode_task_modes`, the web runtime
page) the constraint back from any descriptor.

`census.py` walks the Arena tree and fails on any `declare_parameter`, `ROSParam`,
`rosparam.get(name, default)`, `ParameterDescriptor` or `declare_safe` outside this
module, and on any `Param` or launch argument without a description. It runs as the
`param-census` pre-commit hook and in this package's tests.

`transforms.ThreadedTransformListener` feeds a `tf2_ros.Buffer` from its own
node and executor thread, so `/tf` traffic never wakes the owning node's
executor.

## Spin and lifecycle

`spin.py` owns process-level startup and teardown.

`run_main(node_cls, *args, **kwargs)` calls `asyncio.run(async_main(...))`.
`async_main` initializes rclpy, creates the executor from `create_executor()`
(`rclpy.experimental.EventsExecutor`), spins it in a thread-pool worker,
installs `SIGINT`/`SIGTERM` handlers, calls `node.setup()`, and on shutdown
awaits `node.teardown()` (5 s timeout), cancels pending tasks, drains launches,
and calls `rclpy.try_shutdown()`. A node that decides to exit calls
`self.request_shutdown(reason)` from any thread to take the same path. Calling
`rclpy.try_shutdown()` directly invalidates the context while tasks still run.
After a signal or a request, `SIGINT` stays ignored until the process exits.
A shutdown that has not finished 120 s after it began dumps every thread to
stderr and exits the process with status 1.

`spin_node(node)` is the simpler sync equivalent for non-async nodes, and
`spin_node(node, executor=create_executor())` runs one on the events executor.
On `SIGINT`/`SIGTERM` it shuts the executor down from a helper thread, destroys
the node, then shuts rclpy down. Its own shutdown and the helper's run one at a
time. After a signal, `SIGINT` stays ignored.
`spin_context()` is a context manager that suppresses `KeyboardInterrupt` /
`ExternalShutdownException` and shuts down the executor and rclpy on exit.

Optional `aiomonitor` integration: override `aiomonitor_config()` on the node
to return kwargs for `aiomonitor.start_monitor`, then pass `aiomonitor=True`
to `run_main`.

## Testing

`testing.py` holds the spin helpers ROS-tier tests share. `spin_until(rclpy,
nodes, predicate, timeout_sec)` spins nodes round-robin until the predicate holds
and raises `AssertionError` at the deadline. `spin_in_background(rclpy, node,
stop)` spins one node on a daemon thread until the event is set. It imports rclpy,
so test modules load it through `pytest.importorskip`.
