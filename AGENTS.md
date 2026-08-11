# Arena-Rosnav - Agent Context

If `CLAUDE.md` is present in this directory, read it first. It is the
authoritative project map and rule set and applies verbatim (where it names a
specific agent, read that as "you"). This checkout currently may not contain
that file, so use the per-package READMEs and the repository tree as the source
of truth when it is absent.

## Working memory

Accumulated project knowledge lives in `.codex/arena-memory/`. Start with
`MEMORY.md` (a one-line index) and open the linked files relevant to the task
before exploring the code. Files prefixed `feedback_` are binding workflow
rules. Files prefixed `project_` record technical findings, known bugs, dead
ends, and invariants. Trust them over first impressions, but verify stale file
and flag references against the current tree before acting on them.

After a context compaction, re-read this file and `CLAUDE.md` when present, then
re-open the memory files relevant to the current task.

## Running commands

ROS and simulator commands are intended to run inside the Arena dev container.
Do not run them directly on a host that lacks the configured Arena environment.

- Inside the dev container, run ROS and Arena commands from the workspace root:
  `cd /home/arena/arena_ws && source arena -c '<command>'`. Never run them bare.
- DDS cold cache: the first `ros2 topic list` or `ros2 node list` after startup
  is often empty. Run it twice and trust the second result.
- The container shares a PID namespace. After finishing runs, stop the processes
  you started.
- Never launch Isaac Sim while another instance is running. Check memory
  headroom first and abort if swap use exceeds 1 GB or available RAM is below
  15 GB.

## Git

- Never add AI attribution to commits or pull requests. Do not add
  `Co-Authored-By` or `Generated with ...` trailers.
- Use short, lowercase, imperative commit messages matching repository history.
  Use a subject line only unless asked otherwise.
- Never push or run another remote-mutating command unless the current user
  message explicitly requests it. Broad local authority does not authorize
  remote changes.
- Never stage or commit on your own initiative. The index belongs to the user.
