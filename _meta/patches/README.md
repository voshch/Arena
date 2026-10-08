# Post-update patches

`arena update` runs every script in this directory that has no `<name>.done` beside it and writes that marker when the script exits 0. When the Arena pull moves the checkout, the update reruns itself on the pulled code, so a patch runs in the same update that pulls it. Patches run right after the Arena pull and submodule checkout, before the `.repos` imports, feature updates, rosdep and python deps, also with `GIT=0`. Markers are gitignored, so each checkout applies each patch once. `git clean -x` and `git stash -a` drop them, and the patches run again.

- Executable, any language. It runs unattended (stdin is `/dev/null`) with the Arena checkout as the working directory and the environment of the update (`ARENA_WS_DIR`, `ARENA_DIR`, ...).
- Idempotent: a lost marker or a failed run means another run. Exit non-zero whenever the job is not fully done, so the next run can finish it.
- A non-zero exit lists the patch under "update incomplete" and retries it on the next update.
- Editing a patch does not run it again. Rename it instead.
- Name it `YYYY-MM-DD-<slug>`. Patches run in name order. Dotfiles, directories and `*.md` are not patches.
- A dotfile is a parked patch, written ahead of the change it cleans up. Rename it to `YYYY-MM-DD-<slug>` in the commit that makes it due.
- Delete a patch once no checkout can still need it.
