---
name: feedback_no_claude_in_repo
description: "never let any claude/CLAUDE.md reference enter tracked repo files, ever"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: f0263fb2-f253-42ec-9ecf-e6795abbd2de
---

No "claude" anywhere in the tracked repo, ever: no `CLAUDE.md` committed, no Claude/Anthropic strings in tracked source/docs, no attribution trailers in commits or PRs (see also the global git rule).

**Why:** the user reacted sharply ("no claude in the repo ever!") to CLAUDE.md being touched. `CLAUDE.md` is globally gitignored via `/home/arena/.gitignore` (enforced in superproject and every submodule), so editing it is local-only and harmless, but anything claude-flavored must stay out of tracked files and history.

**How to apply:** edit `CLAUDE.md` freely (it is ignored) but never `git add` it or propose committing it. Before staging any change, grep the staged diff for `claude` (case-insensitive) in every affected repo and confirm CLEAN. Never author claude/Anthropic references into tracked docs or code. Relates to [[feedback_commit_messages_verbatim]] and [[feedback_no_auto_commits]].
