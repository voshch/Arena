# arena_bringup Python helpers

The `arena_bringup` Python package provides launch-time primitives used
throughout `arena_bringup/launch/`. None of these are ROS nodes - they are
pure launch-system utilities imported at Python-load time.

## actions.py

[`actions.py`](actions.py)

### `IsolatedGroupAction`

A `GroupAction` subclass that wraps its children with
`PushLaunchConfigurations` / `PopLaunchConfigurations` and
`PushEnvironment` / `PopEnvironment`. Any launch configuration or environment
variable set inside the group does not leak to the parent scope.

Typical use in `arena_runtime.launch.py` - the simulator is wrapped so its arg mutations are scoped:

```python
IsolatedGroupAction([launch_simulator])
```

## substitutions.py

[`substitutions.py`](substitutions.py)

### `SelectAction`

A `launch.Action` that holds a `str -> list[Action]` registry and executes
only the actions registered under the key that matches a substitution at
launch time.

```python
sel = SelectAction(LaunchConfiguration('sim'))
sel.add('gazebo', IncludeLaunchDescription(...))
sel.add('dummy', GroupAction([]))
# sel.keys == ['gazebo', 'dummy']
```

Used in `sim.launch.py` and `human.launch.py` as the simulator-dispatch
mechanism. The `choices` of the corresponding `LaunchArgument` are set from
`sel.keys` so invalid values are caught at launch-argument validation time.

### YAML substitution helpers

| Class | Purpose |
|---|---|
| `YAMLFileSubstitution` | Reads a YAML file at perform-time, writes it to a tempfile, and returns the path. Supports a `default` dict fallback and optional intra-YAML `${}` substitution. |
| `YAMLRetrieveSubstitution` | Retrieves a nested key from a `YAMLFileSubstitution` using `/`-separated path segments. |
| `YAMLMergeSubstitution` | Deep-merges multiple `YAMLFileSubstitution` objects and returns a tempfile path to the merged result. |
| `YAMLReplaceSubstitution` | Applies `${}` template substitution from a substitutions YAML onto a target YAML, returning the rendered tempfile path. |
| `CurrentNamespaceSubstitution` | Returns `ros_namespace` from the current launch context (or `/` if absent). |

### `_YAMLReplacer`

Internal recursive `${key}`, `${*list}`, `${**dict}`, and `${key:-default}`
substitution engine used by `YAMLReplaceSubstitution`. Not part of the public
API.

## `LaunchArgument`

[`arena_rclpy_mixins.launch_argument`](../../utils/arena_rclpy_mixins/arena_rclpy_mixins/arena_rclpy_mixins/launch_argument.py)
extends `DeclareLaunchArgument` with a mandatory description and:

| Property / method | Returns | Purpose |
|---|---|---|
| `.substitution` | `LaunchConfiguration(name)` | Inline substitution reference |
| `.dict` | `{name: substitution}` | Splat into `launch_arguments=` dicts |
| `.param(type_)` | `{name: ParameterValue(..., value_type=type_)}` | Typed ROS parameter |
| `.str_param` | `{name: ParameterValue(..., str)}` | Shorthand string param |
| `LaunchArgument.from_param(param, prefix=..., keep_default=...)` | `LaunchArgument` | The argument of a parameter: empty default (the node default) when the launch file only forwards it, the parameter default with `keep_default` when the launch file reads it too |

`forwarded(context, params)` gives a node the set arguments of params as typed values.
`declared_arguments(package, file)` loads a launch file and returns the name,
description, default and choices of every argument it declares.

`LaunchArgument.auto_append(ld_items)` - call once at the top of a launch
file; subsequent `LaunchArgument(...)` calls append themselves automatically
to `ld_items`, so every argument declaration appears in the description.

Typical use:

```python
LaunchArgument.auto_append(ld_items)
sim = LaunchArgument(name='sim', default_value='gazebo')
# sim is both declared and appended to ld_items
```

## future.py

[`future.py`](future.py)

### `PythonExpression`

Drop-in replacement for `launch.substitutions.PythonExpression` that
evaluates an arbitrary Python expression string (with `math` symbols available
by default). Accepts a list of mixed `str` / `Substitution` fragments that
are concatenated before `eval`.

Typical use in launch files for derived defaults:

```python
mobile = LaunchArgument(
    name='mobile',
    default_value=PythonExpression(
        [str({"dummy": "none"}), '.get("', sim.substitution, '", "nav2")']
    ),
)
```

### `IfElseSubstitution`

Returns one of two substitution branches depending on a bool-typed condition
substitution:

```python
IfElseSubstitution(headless.substitution, " -s", "")
```

## extensions/NodeLogLevelExtension.py

[`extensions/NodeLogLevelExtension.py`](extensions/NodeLogLevelExtension.py)

### `NodeLogLevelExtension`

A `NodeActionExtension` plugin that reads a JSON-encoded list of
`(pattern, level)` rules from `context.launch_configurations
['NodeLogLevelExtension_log_level']`, matches each `Node`'s FQN against the
rules in first-match-wins order, and prepends `--log-level <level>` to that
node's command-line arguments. Registered as a `launch_ros` plugin via the
`NodeActionExtension` extension point.

### `SetGlobalLogLevelAction`

A `launch.Action` that parses a `log_level` spec (bare scalar, inline `{...}`
rule set, inline `+[...]` / `[...]+` merge form, or YAML file path), merges or
replaces the rules currently in the launch context, and writes them back as
JSON. See the [launch README](../launch/README.md#log-level) for the full
input grammar.

Typical use in `arena_runtime.launch.py`:

```python
SetGlobalLogLevelAction(log_level.substitution)
```

Layering example (used by the task_generator robot launcher):

```python
SetGlobalLogLevelAction(current_log_level)                    # base
SetGlobalLogLevelAction('+[**/controller_server:error, ...]') # prepend overrides
```
