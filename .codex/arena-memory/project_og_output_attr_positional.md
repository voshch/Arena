---
name: OG output dynamic attrs need positional port_type
description: In arena_isaac OmniGraph helper, creating an OUTPUT dynamic attribute on a ScriptNode requires passing AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT as positional 4th arg to og.Controller.create_attribute, not as port_type=... kwarg.
type: project
originSessionId: baf043f8-7a6e-4226-959e-eb2fcac51e83
---
In arena_isaac's `isaac_utils/graphs/__init__.py`, `_Node.create_attribute` auto-detects `outputs:` prefix and calls `og.Controller.create_attribute(path, name, type, og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT)` with the port type **positional**. Passing it as `port_type=...` kwarg gets silently ignored on this OG version (1.141.x), the attribute is created as INPUT, and downstream connects fail with "Parsed source ... as a path attribute, which cannot connect to a destination of type double[]".

**Why:** Spent a debug iteration on it during mecanum drive work. The OG Python wrapper around the C++ binding doesn't honor the `port_type` kwarg name (some docs call it `attribute_port`); positional dodges the question entirely.

**How to apply:** Any new OmniGraph node in arena_isaac that needs a dynamic output attr (ScriptNode in particular) should use the existing `create_attribute('outputs:foo', 'double[]')` helper, which routes through the positional path. If you ever bypass the helper and call `og.Controller.create_attribute` directly, pass port_type positionally.
