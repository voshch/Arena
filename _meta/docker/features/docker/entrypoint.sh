#!/bin/bash

export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
command -v pyenv > /dev/null 2>&1 && eval "$(pyenv init - bash)"

if [ ! -f /.built ]; then
    exec 9> /tmp/arena-first-boot.lock
    if flock -n 9; then
        (
            set -e
            cd /opt/arena_ws
            source ./source
            _arena_venv_provision
            arena resource
            arena registry add docker
            echo "Running initial setup..."
            arena update
            rm -rf build/arena_models install/arena_models
            BUILD_ALL=1 arena build || true
            sudo touch /.built
            echo 'Initial setup complete.'
            echo -e '\033[0mRun \033[01;33marena feature docker commit\033[0m to save this state.'
        )
    elif [ -t 0 ]; then
        echo "arena: initial setup running in another session, shell is ready (docker logs -f for progress)"
    else
        flock 9
    fi
    exec 9>&-
fi

exec "$@"
