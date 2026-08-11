---
name: project-planner-hf-weights-mechanics
description: planner weights live first-order in arena-rosnav HF org; local model files are symlinks into the HF cache with container paths; how to reclaim forwards and publish from the host
metadata: 
  node_type: memory
  type: project
  originSessionId: c83ced68-0743-4e9b-a405-9a300932ea71
---

DRL planner weights are published per planner via `planners/<name>/weights.yaml`: a `files:` list of `{repo, filename, dest, sha256}` where `repo` is the HF model id (`arena-rosnav/<name>`), `filename` is the remote name, `dest` is the local path (`model/<...>`), `sha256` pins content. `model/` and `__pycache__/` are gitignored, so the blobs never go in git, they are fetched from HF.

Weights must live **first-order in the `arena-rosnav` org** (the user explicitly does NOT want `arena-rosnav-shared`). A repo that was moved to another org leaves a **307 forward** at the old `arena-rosnav/<name>` name (check with `GET /api/models/<repo>` `allow_redirects=False`: 200 = first-order, 307 = forward). Reclaim the name first-order with `HfApi.create_repo(repo, repo_type="model", exist_ok=False)`, which shadows the redirect with a fresh real repo, then `upload_file` the blob (the new repo is empty until you do, so always populate immediately).

Local `planners/<p>/model/*` are **symlinks into the HF hub cache using the CONTAINER path** `/opt/arena_ws/data/hf/hub/models--arena-rosnav--<name>/snapshots/<hash>/<file>`. On the host that target dangles (`os.path.realpath` fails); translate the prefix `/opt/arena_ws/` -> `/home/arena/arena_ws/` (`os.readlink` + string swap) to read the real bytes.

Tooling: the host `hf`/`huggingface-cli` shim has a container-baked shebang (`/opt/.../.venv/bin/python3`) and the host `.venv` (py3.14) has no `huggingface_hub`, so both fail on host. Run HF ops on the host via `uvx --from huggingface_hub [--with requests] python -` (or `hf`), which authenticates from the host token at `~/.cache/huggingface/token`. In-container `hf` only has the venv on PATH under the interactive rcfile; the `source ./arena -c '<cmd>'` (`bash --norc`) form does not, so call `/opt/arena_ws/src/Arena/.venv/bin/hf` by full path there.

Verify after upload: `HfApi.get_paths_info(repo,[name]).lfs.sha256` for LFS files; small files store **non-LFS** (no `lfs` object), so confirm those with `hf_hub_download` + `hashlib` instead of treating absent-lfs as a mismatch.

GitHub side: planner repos are public under the `arena-planners` org, default branch `jazzy` (rename `master`->`jazzy` on push). See [[project-deps-vcs-managed]], [[project-planner-peds-namespace]].
