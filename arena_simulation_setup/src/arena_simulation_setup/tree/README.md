# Identifier registry

`arena_simulation_setup.tree` defines the `Identifier` ABC and resolver
hierarchy that all asset types in this package build on. Every shipped asset
type (`ObjectIdentifier`, `WorldIdentifier`, etc.) is a concrete subclass
registered at import time via `.use(resolver, ...)`.

## Core ABC: `Identifier`

[tree/__init__.py:457](__init__.py#L457)

`Identifier[T]` is an `attrs`-based class with a single required field `name: str`.
Calling `.resolve(**kwargs)` walks the class-level resolver list in order,
returning the first non-`None` result from `resolver.resolve(self)`, then calls
`self.load(path, **kwargs)` to produce a `T`.

Entry points, each with a sync twin that spawns a thread with its own event loop:

| Method | Sync twin | Notes |
|---|---|---|
| `await identifier.resolve(**kwargs)` | `resolve_sync(**kwargs)` | Fetches (if needed) and loads; returns `T` |
| `await identifier.probe()` | `probe_sync()` | Walks resolvers calling `locate()`; returns one `ResolverVerdict` per resolver, in order, without transferring anything |
| `await identifier.resolve_write_path()` | `resolve_write_sync()` | Walks resolvers calling `destination()`; returns the first non-`None` write path (`resolve_write_sync()` also loads it via `self.load()`) |

`probe()` never transfers data, so it is cheap to call for diagnostics
(`arena asset find` is built on it). `resolve_write_path()` only ever returns a path
handed back by a resolver's `destination()`, never a cache hit or a network fetch.

Register additional resolvers on a subclass at any time with
`MyIdentifier.use(resolver1, resolver2, ...)`.

## Identifier hierarchy

| Class | File | Adds |
|---|---|---|
| `Identifier[T]` | [tree/__init__.py:457](__init__.py#L457) | `name`, resolver walk, `probe()`/`resolve_write_path()`, `load()` ABC |
| `AssetIdentifier[T]` | [tree/__init__.py:614](__init__.py#L614) | `_asset_type` class var; `relpath()` → `<asset_type>/<name>` |
| `DomainAssetIdentifier[T]` | [tree/__init__.py:637](__init__.py#L637) | `domain` field (default `'Common'`); parse `[domain/]<asset_type>/<name>` |
| `ModifiersDomainAssetIdentifier[T]` | [tree/__init__.py:706](__init__.py#L706) | `modifiers: dict` passed through to `load()`; parse `(str, dict)` tuples |
| `InlineIdentifier[T]` | [tree/__init__.py:767](__init__.py#L767) | Wraps an already-loaded value; `resolve()` returns it directly without disk I/O |

## Resolver hierarchy

Every resolver implements up to three async/sync-callable methods, declared on
`ResolverBase` ([tree/__init__.py:90](__init__.py#L90)):

| Method | Default | Purpose |
|---|---|---|
| `resolve(identifier)` | returns `None` | Read path, fetching over the network if this resolver can |
| `locate(identifier)` | calls `resolve()` | Where the identifier would resolve, without transferring anything |
| `destination(identifier)` | returns `None` | Write path; `None` means "not a write target" |

A concrete resolver overrides whichever of these it answers for. `Identifier.resolve()`
walks `_resolvers` calling `resolve()`. `probe()` walks it calling `locate()`. `resolve_write_path()`
walks it calling `destination()`.

| Class | File | Behaviour |
|---|---|---|
| `SimplePathResolver` | [tree/__init__.py:177](__init__.py#L177) | Fixed `Path`; checks `path / identifier.relpath()`. `locate()` uses the same existence check |
| `DynamicPathResolver` | [tree/__init__.py:191](__init__.py#L191) | `DynamicPath` whose value can change at runtime (e.g. `DynamicPaths.WORLD`); optional transform `fn` |
| `DynamicPathListResolver` | [tree/__init__.py](__init__.py) | `DynamicPathList` whose paths can change at runtime (`DynamicPaths.LINKED`), the first path holding the asset wins; optional transform `fn` |
| `NetResolver` | [tree/__init__.py:212](__init__.py#L212) | Batches same-tick `resolve()` requests; calls `ros2 run arena_assets arena_assets -s net <provider> fetch`. `locate()` instead runs a non-fetching `... exists` check. Constructor takes `formats=` (defaults to the `ARENA_MODELS_FORMATS` env filter; pass `formats=()` to opt out) and `ttl=` (defaults to `ARENA_MODELS_TTL`, seconds). `listall(network=True)` caches the bucket listing at `ARENA_ASSETS_DIR/<provider>/.listing` for `ARENA_MODELS_TTL` regardless of the payload `ttl=`, and `listall_async` shares one in-flight fetch per provider. `expire(identifier=None)` writes `1` into the `.ttl` of one complete entry, or of every one in the bucket and drops its listings, so the next `resolve()` refetches while an unreachable bucket still serves the old copy (`arena asset expire`). `NetResolver.all(_T, providers=NETWORK_PROVIDERS, ...)` builds one instance per bucket name |
| `LocalResolver` | [tree/__init__.py](__init__.py) | Reads and writes a subdirectory of the local root (`$ARENA_ASSETS_DIR_LOCAL`, looked up at every use, never cached): `resolve()` checks existence, `listall_async()` walks it in a child process at every call, `destination()` is the write target |
| `FallbackResolver` | [tree/__init__.py:406](__init__.py#L406) | Write target only: implements `destination()` and nothing else, so `resolve()`/`locate()` always return `None` and it can never shadow a real source |

### `DynamicPaths` singleton

[tree/__init__.py:62](__init__.py#L62)

| Path | Default value |
|---|---|
| `DynamicPaths.WORLD` | `/dev/null` (set at runtime to the active world directory) |
| `DynamicPaths.LINKED` | empty (set at runtime to the directories of the worlds the active world links levels from) |
| `DynamicPaths.ARENA` | `$ARENA_ASSETS_DIR_LOCAL` or `$ARENA_ASSETS_DIR/local`, looked up at every read until a path is assigned |
| `DynamicPaths.IMPORTED` | `$ARENA_ASSETS_DIR_IMPORTED` or `$ARENA_ASSETS_DIR/_imported` (an empty variable counts as unset), where `arena asset import` writes, looked up at every read until a path is assigned |

`DynamicPaths.as_resolvers(T)` returns four resolvers for a given `Identifier`
type: `WORLD / 'assets'`, the `assets/` of every `LINKED` world, `ARENA`, then
`IMPORTED`.

`point_world_assets(path, linked)` ([World/view.py](World/view.py)) points
`WORLD` at `path` (`/dev/null` for None) and `LINKED` at `linked`, then
invalidates every `DynamicPathResolver` and `DynamicPathListResolver`
registered on any `Identifier` kind, so no kind keeps answers cached from the
previous world. The task generator's world load and `SoundLibrary.use_world`
call it. `world_assets(view, world=None)` is the scoped form: a context
manager that points the paths at `view` and the worlds it links levels from
(taken from `world` when already loaded) and restores both on exit, also on
an exception. Offline tools that resolve a world's assets (`nav_at_r`,
`preload_world`, the scenario generator) use it.

## Shipped Identifiers

| Identifier | File | Base class | `_asset_type` | Resolvers (in order) |
|---|---|---|---|---|
| `ObjectIdentifier` | [tree/assets/Object.py](assets/Object.py) | `DomainAssetIdentifier[ObjectView]` | `Object` | DynamicPaths (world/assets + local), then NetResolver |
| `HumanIdentifier` | [tree/assets/Human.py](assets/Human.py) | `DomainAssetIdentifier[HumanView]` | `Human` | DynamicPaths (world/assets + local), then NetResolver |
| `AnimationIdentifier` | [tree/assets/Animation.py](assets/Animation.py) | `DomainAssetIdentifier[AnimationView]` | `Animation` | DynamicPaths (world/assets + local), then NetResolver (`formats=()`) |
| `MaterialIdentifier` | [tree/assets/Material.py](assets/Material.py) | `ModifiersDomainAssetIdentifier[Material]` | `Material` | DynamicPaths (world/assets + local), then NetResolver |
| `SoundIdentifier` | [tree/assets/Sound.py](assets/Sound.py) | `DomainAssetIdentifier[SoundView]` | `Sound` | DynamicPaths (world/assets + local), then NetResolver (`formats=()`, listed by `--children` under `Common/Sound`) |
| `WallIdentifier` | [tree/Wall.py](Wall.py) | `DomainAssetIdentifier[WallDescription]` | `Wall` | DynamicPaths (world/assets + local), then NetResolver |
| `WorldIdentifier` | [tree/World/view.py](World/view.py) | `Identifier[WorldView]` | - | `ComposedResolver`, `ARENA_WORLD_PATH` roots, `LocalResolver('worlds')` (also the write target), `ASS_DIR / 'worlds'` (existence-checked), then a `NetResolver` per `WORLD_BUCKETS` provider |
| `EnvironmentIdentifier` | [tree/configs/environment.py](configs/environment.py) | `Identifier[EnvironmentDescription]` | - | `EnvironmentResolver` → `ASS_DIR / 'configs' / 'environment'` |
| `ParametrizedIdentifier` | [tree/configs/parametrized.py](configs/parametrized.py) | `Identifier[ParametrizedConfig]` | - | `ParametrizedResolver` → `AB_DIR / 'configs' / 'parametrized'` |

### `ObjectIdentifier` / `HumanIdentifier`

`.load()` wraps the resolved path in an asset view (`ObjectView` /
`HumanView`) whose `.model` property is a `ModelWrapper`.
`ObjectIdentifier` provides both SDF and USD providers; `HumanIdentifier`
provides SDF only.

### `AnimationIdentifier`

A pedestrian animation clip: `<domain>/Animation/<name>/` holds `annotation.yaml` (fps, loop, reverse, joint
mask, provenance note; it is also the bucket's listing sentinel) and `<name>.anim`. `.load()` returns an
`AnimationView`: `.meta` is the annotation, `.clip` an `AnimationClip` whose `.frames()` gives the per-frame dicts the
players consume. `write_clip(directory, frames, meta)` writes one. Clips never ship in the repo: both local asset dirs
gitignore `*/Animation/`.

A `.anim` is an npz archive, read with `allow_pickle=False`, whose members are named `<group>/<member>`:

| Group | Members |
| --- | --- |
| `anim` | `requires` (the oldest reader that can play the file), `settings` (JSON over `annotation.yaml`) |
| `clip` | `joint_names (J,)`, `angles (T, J)`, `t`, `root_xy_yaw`, `animation_state` |
| `root` | optional: `quat (T, 4)`, `z (T,)` |
| `table` | optional: the baked pointing table, `point_at_template` only |

The format stays compatible by these rules: readers ignore groups and members they do not know, so adding one needs no
version change. A member never changes meaning, new meaning gets a new name, and a breaking change adds a group such as
`clip.2` beside `clip` so an older reader still plays the file. `anim/requires` rises only when no older reader can play
a file, a reader that sees a higher value warns and falls back to `<name>.npz`.

`.meta` layers three sources, each over the one before: the annotation's top level, `anim/settings`, and the annotation's
`anim:` block. Tuning a clip is a one-line `anim:` edit, older readers ignore the block.

The bucket keeps each clip's old `<name>.npz` and `table.npz` for checkouts from before `.anim`. Those files and their
annotations are never replaced. `NetResolver` fetches only `.anim` and `.yaml` for this kind, so one clip downloads once.
`AnimationClip.legacy` marks a clip read from an old local or cached `.npz`, and `AnimationManager` lists those in one
warning. `pack_anim(source, target, settings)` converts an old clip, `settings` holding what a new reader needs on top of
the bucket's annotation.

Scenarios name clips as strings inside humansim agent configs, so `ScenarioView.identifiers()` reads them
out: `referenced_clips()` collects every attention `clip` (a name or `{name, when}`) of the inline `agent:`
entry and of the scenario-local agent file it names (`agent_type: ./caller.yaml`), plus the default clip of
each `interaction:` step that sets none (humansim's `InteractionKind.clip`, skipped without `arena_humansim`).
That puts clips in the world closure, so `arena asset push world` refuses a world whose clips only resolve
locally, and an unreadable agent file raises rather than passing as clip-free.

### `MaterialIdentifier`

Supports a `modifiers` dict. The only handled modifier is `tint` (a CSS color
string); when present, `load()` copies the material tree to a temp directory and
tints every diffuse texture before returning.

Default materials per context: `wall` → `Marble`, `floor` → `Porcelain_Tile_4`,
`door` → `Aluminum_Anodized`
([tree/assets/Material.py:69](assets/Material.py#L69)).

### `SoundIdentifier`

Resolves `domain/Sound/name/` and wraps it in a `SoundView` whose `.manifest`
is the parsed `name.yaml` beside the wav files.
[tree/assets/sound_catalog.py](assets/sound_catalog.py) parses the manifests:
`SoundLibrary` merges the kinds table (`configs/sounds/kinds.yaml`, `kinds_file()`)
with the `kinds:` fragments of local manifests, resolves assets lazily and picks
variants deterministically (`SoundAsset.select`, `selection_seed`). It imports
neither numpy nor scipy, decoding lives in the acoustics backend. The kinds
table and the manifest schema are documented in
[configs/sounds/README.md](../../../configs/sounds/README.md).

### `WallIdentifier`

Resolves `domain/Wall/name/name.yaml` and structures it as a `WallDescription`.
`WallDescription.realize(start, end)` expands the `main` list into
`(Iterable[WallSegment], Iterable[Obstacle])`.

In `world.yaml`, a wall entry without a `kind` field generates a plain
`PlaceWallSegmentAsset`; a wall entry with `kind: <name>` is resolved via
`WallIdentifier(name)` at simulation time by
[shared/walls.py:25](../shared/walls.py#L25).

### `WorldIdentifier`

Resolvers, in order (first hit wins): the `ComposedResolver` for stacked
worlds (`a,b[1]`), one `SimplePathResolver` per `ARENA_WORLD_PATH` root
(colon-separated env var), then a `LocalResolver` on
`$ARENA_ASSETS_DIR_LOCAL/worlds`, then a `SimplePathResolver` rooted at
`ASS_DIR / 'worlds'`, then one `NetResolver` per `WORLD_BUCKETS` provider
(default `arena-worlds-prod-public`, constructed with `formats=()` since a
world payload is not model files, so the `ARENA_MODELS_FORMATS` filter would
otherwise drop it). `resolve_write_sync()` returns the `LocalResolver`
destination, so generated and authored worlds land in the local root, outside
the install tree, and a local world shadows a shipped one of the same name.

`.load()` returns a `WorldView`, not the parsed YAML. Call
`.load()` on the view to get a `World` (format 2, format 1 upgraded). `WorldIdentifier` has
value equality (`__eq__`/`__hash__` on `name`) and no bespoke `listall`. It
uses the base `Identifier.listall()` aggregation across its resolvers.

### `EnvironmentIdentifier`

Resolved by `EnvironmentResolver` which looks under
`ASS_DIR / 'configs' / 'environment'`. `.yaml` suffix is required; `shortname`
strips it. `EnvironmentDescription` is a plain `dict` subclass (schema is
environment-dependent and not strongly typed).

### `ParametrizedIdentifier`

Resolved from `AB_DIR / 'configs' / 'parametrized'` (the `arena_bringup`
package). Files are XML with a `<random>` root element. `.load()` returns a
`ParametrizedConfig` with three lists: `STATIC`, `INTERACTIVE`, `DYNAMIC`,
each a list of `ObstacleConfig(min, max, type, model, zone)`. `zone` is the
optional zone name that limits placement, empty to place anywhere.

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
