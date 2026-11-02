[![Discord](https://img.shields.io/badge/Discord-Join%20chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/GNTTf9DKyp)
[![Docs](https://img.shields.io/badge/Docs-Read%20online-8CA1AF?logo=readthedocs&logoColor=white)](https://arena-dev.readthedocs.io/)

# Arena-Rosnav

A modular ROS 2 (Jazzy) platform for researching and benchmarking autonomous robot navigation in 2D and 3D simulated environments. It supports classical planners (Nav2), deep-RL planners ([rosnav_rl](https://github.com/Arena-Rosnav/rosnav-rl)), and a variety of simulators (Gazebo, Isaac Sim, MuJoCo, flatsim).

Read [CONTRIBUTING.md](CONTRIBUTING.md) before you open a pull request.

---

## Installation

Prerequisites: [Docker](https://docs.docker.com/engine/install/) or [Podman](https://podman.io/docs/installation) installation with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU support. With Docker, the current user must be in group `docker`. With Podman, install the `podman-docker` alias package and enable the API socket (`systemctl --user enable --now podman.socket`, or `sudo systemctl enable --now podman.socket` when rootful), and for GPU support generate the CDI spec once with `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`.
Afterwards, run the following commands to install Arena:

### Basic Installation

```sh
curl https://raw.githubusercontent.com/voshch/Arena/jazzy/install.sh > install.sh
bash install.sh
```
and follow the prompts. This will create a ROS 2 workspace at your target location and instruct you how to proceed (yellow text).


### Optional Features
```sh
cd ~/arena_ws # replace with your actual workspace path
source arena
arena feature isaac install # optional
arena feature gazebo install # optional
arena feature mujoco install # optional: MuJoCo simulator (sim:=mujoco)
arena feature flatsim install # optional: 2D simulator without a renderer (sim:=flatsim)
arena feature meshnav install # optional: 3D mesh navigation over stairs and escalators (robot.mobile:=meshnav)
arena feature training install # optional
arena feature auditory install # optional: sound simulation (acoustics:=arena)
arena feature hearing install # optional: robot hearing (robot.hearing)
arena feature vllm install # optional: local LLM backend
arena feature docker gpu on # optional: NVIDIA GPU passthrough, needed for training
arena settings net lan # optional: let ROS 2 traffic leave this host
arena settings sim # optional: shows or sets the simulator a launch without sim:= uses
```

We recommend installing at least one simulator. A launch without `sim:=` uses `arena settings sim`, by default the first installed simulator in `SIM_ORDER` ([defaults.py](arena_bringup/arena_bringup/defaults.py)), or `dummy` when none is installed.

GPU passthrough takes effect on the next `source arena`, which recreates the container.

ROS 2 traffic stays on loopback by default (`arena settings net local`), which keeps multi-interface hosts from flooding discovery. `arena settings net lan` opens every interface, Gazebo's own transport always stays local; new shells pick it up on `source arena`, running nodes keep theirs until restarted. The scope lives in the profiles `_meta/docker/fastdds.<net>.xml` and `cyclonedds.<net>.xml`, so `source arena` sets `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`. Do not export `LOCALHOST` after it: rmw_fastrtps then adds a shared-memory transport to every process, which fills a fork's 64 MB `/dev/shm` and leaves segments behind when processes are killed. The Fast DDS profiles probe 120 participant ids on unicast, so a domain with more than 32 processes still discovers under `LOCALHOST`.

#### vllm

Runs a local vLLM server plus a LiteLLM proxy that speaks the Gemini API, so GPT consumers in `task_generator` transparently hit local inference instead of Google. Defaults target an 11 GB 2080 Ti (Qwen3-0.6B, 40% GPU util).

Tune via [`_meta/docker/features/vllm/config.yaml`](_meta/docker/features/vllm/config.yaml):

| key | default | purpose |
| --- | --- | --- |
| `model` | `Qwen/Qwen3-0.6B` | HF model id |
| `gpu_memory_utilization` | `0.4` | fraction of VRAM vllm may claim |
| `max_model_len` | `4096` | context window |
| `port` / `proxy_port` | `8000` / `4000` | vllm / LiteLLM ports |

After editing, re-run `arena feature vllm update` to recreate the container.
The container will start automatically on source and continue running in the background. To free up GPU memory, stop it with `arena feature docker stop`.

## Usage

```sh
cd ~/arena_ws # replace with your actual workspace path
source arena
arena launch sim:=isaac                                         # Isaac Sim
arena launch sim:=mujoco                                        # MuJoCo
arena launch sim:=flatsim                                       # flatsim (2D, no renderer)
arena launch robot.mobile:=rosnav_rl robot.mobile.agent:=<your_agent>       # rosnav_rl DRL planner
arena launch robot.mobile:=drl robot.mobile.planner:=drlvo                  # arena_planners DRL bridge
arena train sim:=gazebo robot.mobile:=rosnav_rl train_config:=<config.yaml>  # DRL training
```

### DRL quick-start
Place your trained agent folder inside `Arena/arena_training/agents/<agent_name>/` (must contain `training_config.yaml` and `best_model.zip`), then launch with `robot.mobile:=rosnav_rl robot.mobile.agent:=<agent_name>`. Refer to the [arena_training](arena_training/README.md) for training instructions.

### arena_planners bridge
For research planners (DRL-VO, CrowdNav, ...) where the policy lives in its own venv, use `robot.mobile:=drl robot.mobile.planner:=<name>`. Install a planner with `arena feature planners add <name>`. The [arena_planners](arena_planners/README.md) submodule handles the bridge, observation pipeline, and HF weight fetch. Optional global plan via `robot.mobile.global_planner:=nav2/navfn`.

### Forks
A fork is an isolated container on a snapshot of your dev tree, with its own ROS domain, so you can edit, build and launch there without touching your checkout.

```sh
source arena --fork [<name>]               # enter a fork, forking the dev tree into it if new (default: lowest free pN)
arena fork new [<name>]                    # the same without entering it
source arena --fork <name> --code          # open it in VS Code instead
arena fork ls                              # list forks
arena fork down <name>                     # delete a fork and its edits
arena evaluation benchmark ... --lanes 4   # one benchmark run across 4 lanes, each in a pool fork
```

`ARENA_FORK_CPUS`, `ARENA_FORK_MEM` and `ARENA_FORK_DOMAIN_BASE` (default 20) in `.env` tune forks. A new fork takes the lowest domain from that base up that no fork holds, no running container sets and no ROS process on the host has bound (any UDP port in the domain's RTPS range).


## Development

### Linting

Linting is handled by [Ruff](https://docs.astral.sh/ruff/), driven by [pre-commit](https://pre-commit.com/). Config lives in root [`pyproject.toml`](pyproject.toml); the hook pin is in [`.pre-commit-config.yaml`](.pre-commit-config.yaml). Formatting is enforced: the `ruff-format` hook fails on any file `ruff format` would change and prints the diff, across the metarepo and its checked-out submodules.

**One-time setup:**
```bash
pip install pre-commit
pre-commit install
```

**Everyday use:** hooks run automatically on `git commit` against staged files. To run manually:
```bash
pre-commit run            # staged files only
pre-commit run -a         # entire repo
ruff check .              # check without pre-commit
ruff format .             # apply the formatting the hook enforces
```

If the hook auto-fixes something, the commit is aborted and the fixes are left unstaged, `git add` and re-commit.

### Type checking

[basedpyright](https://docs.basedpyright.com/) checks types with the `[tool.pyright]` config in the root [`pyproject.toml`](pyproject.toml). It runs inside the Arena container as a manual pre-commit hook:
```bash
pre-commit run basedpyright --hook-stage manual
```

[`_meta/typings`](_meta/typings) holds stubs for compiled modules that ship no type information (USD, MuJoCo, Box2D, Warp, gz-transport, tf2, the Qt bindings).

[`.github/workflows/tests.yml`](.github/workflows/tests.yml) runs the same check, at the hook's basedpyright version, over `_meta` and every package it tests, after the test suites.

### Python dependencies

Each Arena package declares the pip packages it imports in the `[project]` table of its own `pyproject.toml`. Versions and dependencies live only there, `setup.py` keeps the ament glue. `arena update` composes the packages present in your checkout into one uv workspace (`.uv-workspace/`, generated) and syncs it into the venv. Every repo tracks a `uv.lock` derived from the full-tree lock, so a submodule such as `arena_planners` reproduces the same versions standalone with `uv sync`.

Locks follow manifest edits on their own: a commit that stages a `pyproject.toml` or `setup.py` relocks and stages the regenerated `uv.lock`. In this repo that runs through pre-commit, in submodules through `core.hooksPath`, which `arena update` sets. Relocking needs a full tree (all submodules initialized) and `uv` on `PATH`. Without them the hook skips and the `uv-workspace` check flags the stale lock. Locks the hook touched in other repos are listed for you to commit there.

To move versions on purpose, on a full tree:
```bash
python3 _meta/tools/uv_workspace.py lock --upgrade-package NAME   # one package
python3 _meta/tools/uv_workspace.py lock --upgrade                # everything
```
then commit the changed `uv.lock` files, submodules first.

### CI

The lint, tests and coverage workflows run on every push to `jazzy` and every pull request targeting it. Each checks out the change merged into `jazzy` with [`voshch/ci/trial-merge`](https://github.com/voshch/ci): conflicting submodule pins are merged at every depth, and only a conflict outside them fails the run. GitHub starts no `pull_request` run while a pull request has conflicts, so run a conflicting one by hand with `gh workflow run <lint|tests|coverage>.yml --ref jazzy -f pr=<N>`.

[`.github/workflows/lint.yml`](.github/workflows/lint.yml) runs the same pre-commit hooks on that merge, and on each new commit as a separate `lint / per commit` status. The GH check uses the exact config and hook pins from `.pre-commit-config.yaml`, so local and CI never drift. Make the check required in branch protection to block merges on lint failures.

Bump the Ruff version with `pre-commit autoupdate`.

[`.github/workflows/tests.yml`](.github/workflows/tests.yml) runs every test suite of the Arena-owned packages in a `ros:jazzy` container. It installs what they need the way `arena update` does (`rosdep` from each `package.xml`, `colcon --packages-up-to`, `uv_workspace.py compose` with `uv sync`, the default robot's submodules), so a missing `package.xml` or `pyproject.toml` dependency fails it, and runs them with `colcon test`, the runner behind `arena test`, one package per worker on its own `ROS_DOMAIN_ID`.

The container is this or last week's `ghcr.io/voshch/arena-ci:jazzy-<ISO week>` when one exists, else plain `ros:jazzy`. A push to `jazzy` without this week's image dispatches [`.github/workflows/ci-image.yml`](.github/workflows/ci-image.yml), which runs [`.github/ci/provision.sh`](.github/ci/provision.sh) on a clean `ros:jazzy` and publishes the result. The image records the rosdep keys it was provisioned for, and [`.github/ci/system-deps.sh --purge-dropped`](.github/ci/system-deps.sh) purges what only a since-dropped key installed before installing the current ones, so a run in the image installs the same set as a run in plain `ros:jazzy`. Build objects go through ccache and downloads through the uv cache, both content-addressed. The package must be public for runs to pull it.

[`.github/workflows/coverage.yml`](.github/workflows/coverage.yml) gates the changed lines under the `GATED` paths (inside a submodule, the diff between the old and new pin): at least 90% of them must be covered, a partially taken branch counting as uncovered. It builds and tests the packages that own the `GATED` paths, found by [`.github/ci/owning-packages.sh`](.github/ci/owning-packages.sh) from the nearest `package.xml`. Changes outside `GATED` end the job after the diff.

## Troubleshooting

### Unknown runtime specified 'nvidia'

```sh
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart containerd
```

### rviz fails to open / crashes on launch

On hosts with incompatible or missing GPU drivers, rviz can fail to start with an OpenGL error. Force software rendering by adding the following to `.env` at the workspace root:

```sh
LIBGL_ALWAYS_SOFTWARE=1
```

Expect lower framerates, especially with many robots.
