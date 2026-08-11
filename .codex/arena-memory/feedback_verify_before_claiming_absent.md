---
name: Verify before claiming Arena lacks a feature
description: Before saying Arena is missing feature X, grep the codebase for X across multiple plausible names; do not infer absence from MEMORY.md contents
type: feedback
originSessionId: 71d3c70d-8cf7-4c7f-8501-c09b12eb9864
---
When asked "what does Arena not have," never answer from memory or vibes. Grep the codebase for each candidate feature under multiple plausible names before claiming it is absent.

**Why:** User pushed back twice in one conversation — first on `arena feature vllm` (had been called "missing multi-LLM backend"), then on rviz autogen ([utils/rviz_utils/rviz_utils/scripts/rviz_config.py](utils/rviz_utils/rviz_utils/scripts/rviz_config.py) already does it). MEMORY.md is an index of past topics, not a feature inventory — absence from memory ≠ absence from the repo. Asserting absence without checking wastes the user's time and erodes trust.

**How to apply:** For any "does Arena have X / what's missing" question, before the answer goes out: (1) grep the repo for X under 3+ plausible names, including adjacent dirs (`_meta/`, `deps/`, `utils/`); (2) check `_meta/docker/features/` for installable features; (3) only then label something present/partial/absent. If the question is broad, delegate the grep sweep to an Explore agent with an explicit list of search terms per item.
