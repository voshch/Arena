---
name: Don't flag unclear/unexplained intent as an issue
description: In code reviews and quality ratings, do not list "intent unclear", "unexplained", or "would benefit from a comment" as issues. Only flag concrete bugs, real smells, or measurable problems.
type: feedback
originSessionId: d8877697-7174-4a7d-899c-9e2192299c35
---
Do not flag code as having issues merely because its intent is unclear or unexplained. Items like "unexplained asymmetry," "unclear without context," "would benefit from a comment" should not appear in reviews, quality ratings, or plans.

**Why:** The user does not want intent comments added to code and does not consider missing explanations to be a quality issue. Flagging them inflates review lists with non-actionable noise and conflicts with the broader "don't explain WHAT the code does" preference.

**How to apply:** When reviewing code, only list concrete bugs, real smells, dead code, actual correctness issues, or measurable problems (perf, security, fragility). If the only observation is "I don't understand why," either investigate until you do or say nothing. Never propose adding a comment as a fix for unclear intent.
