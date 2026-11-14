# Generators

Every generator here writes plain format 2 worlds and scenarios, so the stock task modes run them. The packages layer, each importing only those above it:

| Package | Holds |
|---|---|
| [`core`](#core) | what every generator shares |
| [`world`](world/README.md) | world generators: parameters to a `World`, its layout and the scenarios it comes with |
| [`scenario`](scenario/README.md) | scenarios for an existing world: zone roles, base populations, prompt-driven cases, suites over them |
| [`egobench`](egobench/README.md), [`usar`](usar/README.md) | studies: a manifest of worlds and scenarios, sampled, placed and indexed |

## Core

| Module | Content |
|---|---|
| `bodies.py` | robot and pedestrian radii and the wall clearance placements keep |
| `axes.py` | a manifest axis: a value, a list or a `Span`, gridded or drawn |
| `generated.py` | `Generated` (scenario, agent types, provenance, cameras), `pack` into a scenario directory, `Rejected`, the episode `binding` and `GENERATED_MARKER` |
| `profiles.py`, `agents.py`, `frames.py` | pedestrian profiles and parameter tiers, the behavior-tree type files beside a scenario, level and map frame shifts |
| `triggers.py` | the `robot within` atom and the release point along a robot route |
