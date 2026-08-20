# Identifier registry

`arena_simulation_setup.tree` defines the `Identifier` ABC and resolver
hierarchy that all asset types in this package build on. Every shipped asset
type (`ObjectIdentifier`, `WorldIdentifier`, etc.) is a concrete subclass
registered at import time via `.use(resolver, ...)`.

## Core ABC: `Identifier`

[tree/__init__.py:376](__init__.py#L376)

`Identifier[T]` is an `attrs`-based class with a single required field `name: str`.
Calling `.resolve(**kwargs)` walks the class-level resolver list in order,
returning the first non-`None` result from `resolver.resolve(self)`, then calls
`self.load(path, **kwargs)` to produce a `T`.

Two synchronous-friendly entry points:

| Method | Notes |
|---|---|
| `await identifier.resolve(**kwargs)` | Async; used in simulator contexts |
| `identifier.resolve_sync(**kwargs)` | Spawns a thread with its own event loop; safe from sync call sites |

Register additional resolvers on a subclass at any time with
`MyIdentifier.use(resolver1, resolver2, ...)`.

## Identifier hierarchy

| Class | File | Adds |
|---|---|---|
| `Identifier[T]` | [tree/__init__.py:376](__init__.py#L376) | `name`, resolver walk, `load()` ABC |
| `AssetIdentifier[T]` | [tree/__init__.py:505](__init__.py#L505) | `_asset_type` class var; `relpath()` → `<asset_type>/<name>` |
| `DomainAssetIdentifier[T]` | [tree/__init__.py:536](__init__.py#L536) | `domain` field (default `'Common'`); parse `[domain/]<asset_type>/<name>` |
| `ModifiersDomainAssetIdentifier[T]` | [tree/__init__.py:617](__init__.py#L617) | `modifiers: dict` passed through to `load()`; parse `(str, dict)` tuples |
| `InlineIdentifier[T]` | [tree/__init__.py:685](__init__.py#L685) | Wraps an already-loaded value; `resolve()` returns it directly without disk I/O |

## Resolver hierarchy

| Class | File | Behaviour |
|---|---|---|
| `SimplePathResolver` | [tree/__init__.py:151](__init__.py#L151) | Fixed `Path`; checks `path / identifier.relpath()` |
| `DynamicPathResolver` | [tree/__init__.py:165](__init__.py#L165) | `DynamicPath` whose value can change at runtime (e.g. `DynamicPaths.WORLD`); optional transform `fn` |
| `NetResolver` | [tree/__init__.py:183](__init__.py#L183) | Batches same-tick requests and calls `ros2 run arena_models arena_models net <provider> fetch`. `listall(network=True)` runs `net <provider> list` and caches the bucket listing at `ARENA_ASSETS_DIR/<provider>/.listing` for `ARENA_MODELS_TTL` seconds (default one day). `listall_async` shares one in-flight fetch per provider |
| `FallbackResolver` | [tree/__init__.py:306](__init__.py#L306) | Always yields `path / identifier.relpath()` without existence check; used for write targets |

### `DynamicPaths` singleton

[tree/__init__.py:53](__init__.py#L53)

| Path | Default value |
|---|---|
| `DynamicPaths.WORLD` | `/dev/null` (set at runtime to the active world directory) |
| `DynamicPaths.ARENA` | `$ARENA_ASSETS_DIR_LOCAL` or `$ARENA_ASSETS_DIR/local` |

`DynamicPaths.as_resolvers(T)` returns two `DynamicPathResolver` instances for a
given `Identifier` type: one pointing at `WORLD / 'assets'` and one at `ARENA`.

## Shipped Identifiers

| Identifier | File | Base class | `_asset_type` | Resolvers (in order) |
|---|---|---|---|---|
| `ObjectIdentifier` | [tree/assets/Object.py](assets/Object.py) | `DomainAssetIdentifier[ObjectView]` | `Object` | DynamicPaths (world/assets + local), then NetResolver |
| `HumanIdentifier` | [tree/assets/Human.py](assets/Human.py) | `DomainAssetIdentifier[HumanView]` | `Human` | DynamicPaths (world/assets + local), then NetResolver |
| `MaterialIdentifier` | [tree/assets/Material.py](assets/Material.py) | `ModifiersDomainAssetIdentifier[Material]` | `Material` | DynamicPaths (world/assets + local), then NetResolver |
| `WallIdentifier` | [tree/Wall.py](Wall.py) | `DomainAssetIdentifier[WallDescription]` | `Wall` | DynamicPaths (world/assets + local), then NetResolver |
| `WorldIdentifier` | [tree/World/World.py](World/World.py) | `Identifier[World]` | - | FallbackResolver → `ASS_DIR / 'worlds'` |
| `EnvironmentIdentifier` | [tree/configs/environment.py](configs/environment.py) | `Identifier[EnvironmentDescription]` | - | `EnvironmentResolver` → `ASS_DIR / 'configs' / 'environment'` |
| `ParametrizedIdentifier` | [tree/configs/parametrized.py](configs/parametrized.py) | `Identifier[ParametrizedConfig]` | - | `ParametrizedResolver` → `AB_DIR / 'configs' / 'parametrized'` |

### `ObjectIdentifier` / `HumanIdentifier`

`.load()` wraps the resolved path in an asset view (`ObjectView` /
`HumanView`) whose `.model` property is a `ModelWrapper`.
`ObjectIdentifier` provides both SDF and USD providers; `HumanIdentifier`
provides SDF only.

### `MaterialIdentifier`

Supports a `modifiers` dict. The only handled modifier is `tint` (a CSS color
string); when present, `load()` copies the material tree to a temp directory and
tints every diffuse texture before returning.

Default materials per context: `wall` → `Marble`, `floor` → `Porcelain_Tile_4`,
`door` → `Aluminum_Anodized`
([tree/assets/Material.py:69](assets/Material.py#L69)).

### `WallIdentifier`

Resolves `domain/Wall/name/name.yaml` and structures it as a `WallDescription`.
`WallDescription.realize(start, end)` expands the `main` list into
`(Iterable[WallSegment], Iterable[Obstacle])`.

In `world.yaml`, a wall entry without a `kind` field generates a plain
`PlaceWallSegmentAsset`; a wall entry with `kind: <name>` is resolved via
`WallIdentifier(name)` at simulation time by
[shared/walls.py:25](../shared/walls.py#L25).

### `WorldIdentifier`

Uses a `FallbackResolver` (no existence check) rooted at
`ASS_DIR / 'worlds'`. `.load()` returns a `World` view object, not the parsed
YAML, call `world.load()` on the view to get a `WorldDescription`.

### `EnvironmentIdentifier`

Resolved by `EnvironmentResolver` which looks under
`ASS_DIR / 'configs' / 'environment'`. `.yaml` suffix is required; `shortname`
strips it. `EnvironmentDescription` is a plain `dict` subclass (schema is
environment-dependent and not strongly typed).

### `ParametrizedIdentifier`

Resolved from `AB_DIR / 'configs' / 'parametrized'` (the `arena_bringup`
package). Files are XML with a `<random>` root element. `.load()` returns a
`ParametrizedConfig` with three lists: `STATIC`, `INTERACTIVE`, `DYNAMIC`,
each a list of `ObstacleConfig(min, max, type, model)`.

## Adding a new Identifier

```python
# my_package/MyAsset.py
import attrs
from pathlib import Path
from arena_simulation_setup.tree import DomainAssetIdentifier, DynamicPaths, NetResolver

@attrs.define(eq=False, hash=False)
class MyAssetIdentifier(DomainAssetIdentifier["MyAsset"]):
    _asset_type = "MyAsset"          # directory name under domain/

    def load(self, path: Path, /, **kwargs) -> "MyAsset":
        ...                          # read files under path/, return a MyAsset

MyAssetIdentifier.use(*DynamicPaths.as_resolvers(MyAssetIdentifier))
MyAssetIdentifier.use(*NetResolver.all(MyAssetIdentifier))
```

Resolvers are checked in registration order. Register fixed-path overrides
first if they should take precedence over network assets.
