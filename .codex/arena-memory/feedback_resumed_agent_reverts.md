---
name: resumed-agent-reverts
description: "A resumed subagent may re-read its original spec, treat later amendments as nonexistent, and revert them"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5ba49527-47d6-4a50-9f6e-500945128630
---

A completed subagent resumed via SendMessage reverted a user-directed amendment (engine velocity sync) because the amendment arrived as a message, not in its original prompt, and it "corrected" the tree back to spec.

**Why:** resumed agents can rebuild intent from the original prompt and dismiss mid-flight amendments as noise, silently undoing verified work.

**How to apply:** after any resume of an agent that owns files, diff the tree against the last verified state before trusting it. Put amendments into the original prompt when possible, and treat a resumed agent's "deviation" reports as changes to audit, not just notes.
