ARG ROS_DISTRO=jazzy
FROM osrf/ros:${ROS_DISTRO}-desktop AS base


RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python-is-python3 \
    python3-pip \
    pipx \
    python3-colcon-common-extensions \
    python3-rosdep \
    libncurses-dev \
    libncursesw5-dev \
    sudo \
    ros-jazzy-rmw-cyclonedds-cpp

RUN PIPX_HOME=/opt/pipx PIPX_BIN_DIR=/usr/local/bin \
    pipx install git+https://github.com/voshch/vcstool.git

WORKDIR /opt/arena_ws

# install python deps
FROM base AS python_deps
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build
COPY src/Arena/pyproject.toml uv.lock* ./

RUN uv venv /opt/venv --system-site-packages


# merge
FROM base AS final
ARG username=arena
ARG group=arena
ARG uid=1000
ARG gid=1000

COPY --from=python_deps --chown=${uid}:${gid} /opt/venv /opt/venv
COPY --from=python_deps /usr/local/bin/uv /usr/local/bin/uv


# install docker cli
RUN  --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
    && install -m 0755 -d /etc/apt/keyrings \
        && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
        && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# install lightsfm
ADD https://github.com/robotics-upo/lightsfm.git /lightsfm
RUN cd /lightsfm && make && make install



# set user perms
RUN (getent passwd ${uid} && userdel -rf $(getent passwd ${uid} | cut -d: -f1)) 2>/dev/null || true && \
    (getent group ${gid} && groupdel -f $(getent group ${gid} | cut -d: -f1)) 2>/dev/null || true && \
    groupadd -g ${gid} ${group} && \
    useradd -m -u ${uid} -g ${gid} -s /bin/bash ${username} && \
    passwd -d ${username} && \
    passwd -d root && \
    groupadd -r sudo || true && \
    usermod -aG sudo ${username}

    
WORKDIR /opt/arena_ws

RUN mkdir -p build
RUN mkdir -p install
RUN mkdir -p log
RUN chown -R ${uid}:${gid} /opt/arena_ws


COPY src/Arena/_meta/docker/features/docker/entrypoint.sh /entrypoint.sh
RUN ln -fs src/Arena/_meta/docker/features/docker/source.container ./source

RUN chmod +x /entrypoint.sh
USER ${username}
RUN curl -fsSL https://pyenv.run | bash

ENTRYPOINT ["/entrypoint.sh"]


# cleanup
# RUN rm -rf /var/lib/apt/lists/*
