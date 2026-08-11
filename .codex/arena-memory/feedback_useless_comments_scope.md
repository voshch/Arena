---
name: Useless-comment rule scope
description: "Useless comments" means tombstones/narration of removals, not docstrings; keep terse class/method docstrings
type: feedback
originSessionId: 5be224a0-4c76-4132-afab-eea4dac5b77d
---
When the user says to drop "useless comments," they specifically mean inline comments that reference what was removed or narrate the change ("# was X before", "# now using Y instead", "// removed Z"), not docstrings. Class and method docstrings, even short ones, are not in scope and should be kept.

**Why:** I overcorrected once by stripping legitimate `"""..."""` docstrings on classes and methods (`_Shelf`, `place`, `_pack`, etc.) along with the actually-targeted removal-tombstone comments; the user clarified docstrings are fine.

**How to apply:** when stripping comments per a "no useless comments" / "no narrating comments" instruction, only target inline `# ...` narration that wouldn't make sense to a cold reader. Keep `"""..."""` docstrings on classes, methods, and helpers unless they're prose blocks that pad without informing.
