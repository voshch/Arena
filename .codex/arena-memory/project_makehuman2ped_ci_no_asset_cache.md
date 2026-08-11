---
name: makehuman2ped-ci-no-asset-cache
description: "makehuman2ped GitHub CI has no MakeHuman asset cache, roster tests fail on missing assets (tongue01)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 26189c30-8f48-467e-852a-a6dc9b883705
---

makehuman2ped's GitHub CI (pytest src/tests/) fails 3 roster tests with "unknown asset name 'tongue01', available: " because the runner has no MakeHuman asset cache (user flagged 2026-07-03: "keep this in mind for github, no cache"). Tests that resolve assets via build_asset_index() need either a vendored minimal asset fixture, a cache-restore step, or skip-when-absent guards before CI can go green. Related: [[makehuman2ped-pipeline]].
