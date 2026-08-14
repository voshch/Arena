#!/bin/bash -i
set -e

export ARENA_REPO=${ARENA_REPO:-https://github.com/voshch/Arena.git}
export ARENA_BRANCH=${ARENA_BRANCH:-jazzy}
export ARENA_ROS_DISTRO=${ARENA_ROS_DISTRO:-jazzy}

read_default(){
    local prompt=$1
    local default=$2
    local result
    
    if [[ -t 0 ]]; then
        read -rp "$prompt [$default]: " result
        echo "${result:-$default}"
    else
        echo "$default"
    fi
}

# == read inputs ==
echo 'Configuring Arena...'

ARENA_WS_DIR=$(realpath "$(eval echo "$(read_default "Arena workspace directory" "${ARENA_WS_DIR:-~/arena_ws}")")")
export ARENA_WS_DIR

arena_lib="$ARENA_WS_DIR/src/Arena/_meta/docker/lib"
# shellcheck source=_meta/docker/lib
if [ -f "$arena_lib" ] && source "$arena_lib" && docker image inspect "$ARENA_IMAGE" > /dev/null 2>&1 ; then
    echo "found an existing image $ARENA_IMAGE for this workspace"
    case "$(read_default 'remove the existing install first? (y/N)' 'N')" in
        y|Y|yes|YES)
            if [ -e "$ARENA_WS_DIR/arena" ] ; then
                ( cd "$ARENA_WS_DIR" && source ./arena uninstall -y ) \
                    || echo 'uninstall failed, continuing' >&2
            else
                echo "no launcher at $ARENA_WS_DIR/arena, skipping uninstall" >&2
            fi
        ;;
    esac
fi

mkdir -p "$ARENA_WS_DIR"
cd "$ARENA_WS_DIR"

# set up
mkdir -p src
if [ ! -d src/Arena ]; then
    git clone "$ARENA_REPO" -b "$ARENA_BRANCH" src/Arena
fi

# shellcheck source=_meta/docker/lib
source "$ARENA_WS_DIR/src/Arena/_meta/docker/lib"
echo "installing ${ARENA_REPO}:${ARENA_BRANCH} on ROS 2 ${ARENA_ROS_DISTRO} as ${ARENA_IMAGE}"
sudo echo 'confirmed'

ln -rsf "$ARENA_WS_DIR/src/Arena/_meta/docker/source" ./arena
ln -rsf "$ARENA_WS_DIR/src/Arena/_meta/tools/Arena.code-workspace" ./ws-arena.code-workspace

echo 'Building Arena...'
cd $ARENA_WS_DIR
printf '%s\n' \
    'arena feature robots install' \
    'arena feature robots add jackal turtlebot mpo700 arm/ur' \
    'exit' \
| source arena

echo 'Installed Arena'
echo 'run the following to get started:'
echo "  cd $ARENA_WS_DIR && source arena"