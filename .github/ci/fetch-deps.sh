#!/bin/bash
set -eo pipefail
git config --global --add safe.directory '*'
vcs import --shallow --recursive --input _meta/repos/arena.repos .
ln -s ../arena_training/deps/rosnav_rl/rosnav_rl deps/rosnav_rl
ln -s ../arena_training/deps/rosnav_rl/rosnav_rl_msgs deps/rosnav_rl_msgs
for path in $(python3 -c 'import pathlib, re, sys; sys.path.insert(0, "_meta"); from arena_cli.features.robots import robot_submodules; robot = re.search(r"_ARENA_DEFAULT_ROBOT = \"(\w+)\"", pathlib.Path("arena_runtime/arena_runtime/arena_runtime/env/robot_manager/robots_manager.py").read_text()).group(1); print(*robot_submodules(pathlib.Path("."))[robot])'); do
  git -C arena_robots submodule update --init --recursive --checkout --depth 1 "${path#arena_robots/}"
done
