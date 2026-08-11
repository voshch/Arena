---
name: No hasattr, no defensive getattr, code passes ruff
description: Write code that passes `ruff check`. No hasattr. No getattr with string-literal attribute name and a default — it's the same anti-pattern.
type: feedback
originSessionId: 8e517d88-2948-4b67-bdb3-656cac73a7d4
---
Write code that passes `ruff check` cleanly. No `# noqa` escape hatches without a specific reason.

Specifically:
- **No `hasattr(obj, 'name')`** as a guard. Access the attribute directly.
- **No defensive `getattr(obj, 'name', default)`** with a string-literal attribute name. `getattr` is only legitimate for truly dynamic dispatch (variable attribute name computed at runtime).
- If the type-checker complains, **fix the type annotation**, not the access. If the attribute is documented/typed to exist, use it.

**Why:** Both `hasattr` and literal-name `getattr` paper over uncertainty about the schema instead of resolving it. They make code brittle to renames and hide real type errors. Ruff's `SIM` family flags them. The user finds this pattern particularly irritating and will call it out if seen.

**How to apply:**
- For fields you expect to exist: write `obj.attr` directly. If static analysis errors, fix the type annotation, not the access.
- For genuinely optional fields declared as such in the type (e.g. `Optional[X]`): use `obj.attr or default`, or just handle `None` explicitly.
- For runtime-discovered attribute names (e.g. iterating a config dict and dispatching by key): `getattr(obj, name_var)` with no default, and let it raise.
- Added to Arena's `CLAUDE.md` under Development so it applies to all future Claude sessions on this repo.
