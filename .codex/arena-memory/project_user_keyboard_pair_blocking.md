---
name: user-keyboard-pair-blocking
description: "User's physical keyboard blocks the Shift+Up and Left+Down key pairs, WASD is immune"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5ba49527-47d6-4a50-9f6e-500945128630
---

The user's keyboard hardware blocks two specific 2-key combos at the matrix level: Shift+Up and Left+Down (second key never reaches X11, user visually confirmed 2026-07-18 with an online keyboard tester). WASD and all other arrow/modifier combos deliver fine. The human_steering panel ships WASD teleop alternates for exactly this reason, and reads Shift via queryKeyboardModifiers().

If a "chord doesn't work" report names one of these pairs, it is the hardware, not app logic. Diagnose new pairs with xev before touching input code, and use a same-code-path alternate binding as the control experiment.
