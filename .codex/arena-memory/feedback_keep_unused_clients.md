---
name: Keep unused service clients pre-instantiated
description: don't drop ROS service clients from adapters just because no call site exists yet, even when refactoring/porting
type: feedback
originSessionId: 74d3ad74-eb35-4967-a58d-011de9f5c02d
---
When auditing an adapter, don't propose removing ROS service `ClientWrapper`s that have no live call sites. Pre-instantiating clients (and `await ensure()`-ing them at setup) is the codebase convention, kept even when the corresponding API isn't used yet.

**Why:** the user wants adapters to advertise the full surface of the underlying service API up front, so adding a call site later is a one-liner and so the setup-time `ensure()` doubles as a health check that the upstream service is alive.

**How to apply:** in adapter reviews / refactors / ports, leave dormant clients in place. Flag them as "dead weight" only if explicitly asked. Same rule applies to imports of the corresponding srv types.
