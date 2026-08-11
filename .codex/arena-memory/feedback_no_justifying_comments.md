---
name: No justifying comments on edits
description: Don't add comments that justify/narrate an edit OR explain how tricky code works; default to zero comments
type: feedback
originSessionId: 1ab8a493-4133-4019-9e49-9d51b9612388
modified: 2026-07-21T20:52:22.092Z
---
Don't leave comments that justify a change ("previously X raced Y", "block here so Z doesn't happen") or tombstone a removal ("this used to do X"). The commit message carries that context.

This extends to *explanatory* comments on tricky logic, not just change-narration. A single-line "why this math works" comment (e.g. `# row 0 is the top of the raster, so y is flipped`) on a coordinate transform was flagged sarcastically as "ailike comments yay." The code is expected to stand on its own.

**Why:** user repeatedly flagged both change-narration AND explanatory comments as noise. "DO NOT ADD DELETION COMMENTS. WHEN YOU DELETE SOMETHING, YOU'RE NOT LEAVING TRACES AS GARBAGE." Sees AI-authored explanatory comments as a tell/clutter.

**How to apply:** default to ZERO comments on edits. Don't talk yourself into "but this one is a legitimate why-comment" for tricky logic, that still gets rejected here. Terse docstrings are fine (see [[feedback_useless_comments_scope]]); inline explanatory/justifying comments are not.

Same rule for README/doc edits: state the mechanism in one line, no parentheticals explaining the refactor or "thin shim around X" asides ("very verbose readme/comments huh", 2026-06-12, after a justifying launch-file comment plus a two-line README entry with a shim parenthetical).

2026-07-21: left multi-line "why" comments on several unpause/reorder edits ("what's those big ass comments", "if i left every one of those in, the repo would be 50% comments after 3 months"). Trimming to one-liners was NOT the fix, the user wanted them gone entirely. Justifying-an-edit comments have zero long-term value once the diff scrolls past, only the code stays. Default to none.
