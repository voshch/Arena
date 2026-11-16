# Generators

Every generator here writes plain format 2 worlds and scenarios, so the stock task modes run them. The packages layer, each importing only those above it:

| Package | Holds |
|---|---|
| [`core`](#core) | what every generator shares |
| [`world`](world/README.md) | world generators: parameters to a `World`, its layout and the scenarios it comes with |
| [`scenario`](scenario/README.md) | scenarios for an existing world: zone roles, base populations, prompt-driven cases, suites over them |
| [`study`](#study) | a manifest's items generated on their worlds, indexed and bundled as benchmark suites |
| [`egobench`](egobench/README.md), [`usar`](usar/README.md) | studies: a manifest, its sampler and its placer |

## Core

| Module | Content |
|---|---|
| `bodies.py` | robot and pedestrian radii and the wall clearance placements keep |
| `axes.py` | a manifest axis: a value, a list or a `Span`, gridded or drawn |
| `generated.py` | `Generated` (scenario, agent types, provenance, cameras), `pack` into a scenario directory, `Rejected`, the episode `binding` and `GENERATED_MARKER` |
| `profiles.py`, `agents.py`, `frames.py` | pedestrian profiles and parameter tiers, the behavior-tree type files beside a scenario, level and map frame shifts |
| `triggers.py` | the `robot within` atom and the release point along a robot route |

A scenario packed with `marked` carries `GENERATED_MARKER`, so `generate_scenario` may replace or prune it. The scenarios a world generator comes with are packed without it.

## Study

A study is a manifest of items, each naming a world generator, its configuration and seed, and a scenario on that world. `study.runner.run` generates each world once for all its items, asks the study's placer for each item's scenario or the reason it does not fit, and writes a world only when one of its items was placed. A world's key is the study's prefix plus `digest` of what generates it, its seed `seed * 1000 + variant`, so the same manifest always writes the same worlds.

| Module | Content |
|---|---|
| `manifest.py` | `StudyManifest` (`name`, `seed`), world seeds and keys, world axes checked against their generator, the grid of configurations |
| `runner.py` | `Item`, `run`, `Outcome`, `Report` with the built worlds, and the `<name>.index.yaml` it writes |
| `cli.py` | the shared `<manifest> --out --limit --write` command line |
| `bundle.py` | each split as one benchmark suite directory with its worlds, `suite.yaml`, README and checksummed index, swapped in whole and replacing only bundles of the same manifest and seed |

A study supplies its items, its placer and, for bundles, its stages and README.
