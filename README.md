[![Discord](https://img.shields.io/badge/Discord-Join%20chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/GNTTf9DKyp)
[![Docs](https://img.shields.io/badge/Docs-Read%20online-8CA1AF?logo=readthedocs&logoColor=white)](https://arena-dev.readthedocs.io/)

# Arena-Rosnav

A modular ROS 2 (Jazzy) platform for researching and benchmarking autonomous robot navigation in 2D and 3D simulated environments. It supports classical planners (Nav2), deep-RL planners ([rosnav_rl](https://github.com/Arena-Rosnav/rosnav-rl)), and a variety of simulators (Gazebo, Isaac Sim).

---

## Installation

Prerequisites: [Docker](https://docs.docker.com/engine/install/) installation with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU support. Current user must be in group `docker`.
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
arena feature training install # optional
arena feature vllm install # optional: local LLM backend
```

We recommend installing at least one simulator.

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
arena launch robot.mobile:=rosnav_rl robot.mobile.agent:=<your_agent>       # rosnav_rl DRL planner
arena launch robot.mobile:=drl robot.mobile.planner:=drlvo                  # arena_planners DRL bridge
arena train sim:=gazebo robot.mobile:=rosnav_rl train_config:=<config.yaml>  # DRL training
```

### DRL quick-start
Place your trained agent folder inside `Arena/arena_training/agents/<agent_name>/` (must contain `training_config.yaml` and `best_model.zip`), then launch with `robot.mobile:=rosnav_rl robot.mobile.agent:=<agent_name>`. Refer to the [arena_training](arena_training/README.md) for training instructions.

### arena_planners bridge
For research planners (DRL-VO, CrowdNav, ...) where the policy lives in its own venv, use `robot.mobile:=drl robot.mobile.planner:=<name>`. Install a planner with `arena feature planners add <name>`. The [arena_planners](arena_planners/README.md) submodule handles the bridge, observation pipeline, and HF weight fetch. Optional global plan via `robot.mobile.global_planner:=nav2/navfn`.


## Development

### Linting

Linting is handled by [Ruff](https://docs.astral.sh/ruff/), driven by [pre-commit](https://pre-commit.com/). Config lives in root [`pyproject.toml`](pyproject.toml); the hook pin is in [`.pre-commit-config.yaml`](.pre-commit-config.yaml). Auto-formatting is intentionally not enforced.

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
```

If the hook auto-fixes something, the commit is aborted and the fixes are left unstaged, `git add` and re-commit.

### CI

[`.github/workflows/lint.yml`](.github/workflows/lint.yml) runs the same pre-commit hooks on every push to `jazzy` and every pull request targeting it. The GH check uses the exact config and hook pins from `.pre-commit-config.yaml`, so local and CI never drift. Make the check required in branch protection to block merges on lint failures.

Bump the Ruff version with `pre-commit autoupdate`.

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
