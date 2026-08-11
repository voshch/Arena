---
name: feedback_fewer_why_comments
description: "Default to omitting why/explanatory comments, including in package manifests (package.xml, .repos, CMake)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: adb09fd1-2764-4e81-b411-5d8e3587be79
---

Lean toward FEWER explanatory/"why" comments in general, not just in code. This explicitly extends to package/manifest files: package.xml exec_depend blocks, .repos pins, CMakeLists. Do not annotate each dependency with what it provides or why it is pinned, and do not add UNMAINTAINED/version-rationale notes inline. Add a comment only when the context is genuinely non-obvious and load-bearing, and keep it to a terse one line matching the file's existing style (e.g. arena.repos already uses short one-liners).

**Why:** user said "less why comments in general, in packages too" after I larded package.xml/arena.repos/msg files with 2-3 line rationale comments.

**How to apply:** prefer zero comments; the dep name and field name usually self-document. Extends [[feedback_no_justifying_comments]] and [[feedback_useless_comments_scope]] beyond code into manifests. Still keep terse docstrings where those already belong.
