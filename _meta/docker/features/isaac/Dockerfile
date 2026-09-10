# --- Stage 1: Builder (Python 3.12 + ROS Jazzy) ---
# Isaac Sim 6.0 bundles Python 3.12, matching Jazzy's native interpreter, so the
# messages are built against the system python3 with no deadsnakes/venv detour.
FROM docker.io/osrf/ros:jazzy-desktop AS builder

RUN apt-get update && apt-get install -y \
    build-essential cmake git curl \
    python3 python3-dev python3-venv python3-pip \
    python3-colcon-common-extensions && rm -rf /var/lib/apt/lists/*

# STRICT CHECK: Fail build if version is not 3.12 (Isaac Sim 6.0 ABI)
RUN python3 --version && \
    python3 -c "import sys; assert sys.version_info[:2] == (3, 12), 'ERROR: Python version is not 3.12!'"

WORKDIR /opt/msg_build

ADD src/Arena/arena_isaac/isaacsim_msgs src/isaacsim_msgs
ADD src/Arena/utils/msgs/arena_people_msgs src/arena_people_msgs

# Install build dependencies (Including NumPy for the C-extensions)
RUN . /opt/ros/jazzy/setup.sh && \
    python3 -m pip install --break-system-packages catkin_pkg empy==3.3.4 lark numpy

RUN . /opt/ros/jazzy/setup.sh && \
    colcon build \
    --cmake-args \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0 \
    -DPYTHON_INCLUDE_DIR=/usr/include/python3.12 \
    -DROSIDL_GENERATOR_PY_PYTHON_EXECUTABLE=/usr/bin/python3

# --- Stage 2: Final Isaac Sim Image ---
FROM nvcr.io/nvidia/isaac-sim:6.0.0

# 6. Switch to root to fix system libs and paths
USER root

# Install cattrs directly into Isaac's internal site-packages to ensure it persists
RUN /isaac-sim/python.sh -m pip install cattrs pyyaml

# Copy the libpython3.12 library from builder (ABI match for the bundled Isaac 6.0 python)
COPY --from=builder /usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0 /usr/lib/x86_64-linux-gnu/
RUN ldconfig

# Copy the built messages
COPY --from=builder /opt/msg_build/install /opt/isaac_bridge_msgs

# 7. Persistent Environment Setup
# We write to /root/.bashrc because Isaac Sim's launch scripts often wipe 'ENV' variables
RUN echo 'export ROS_DISTRO=jazzy' >> /root/.bashrc && \
    echo 'export ACCEPT_EULA=Y' >> /root/.bashrc && \
    echo 'export PRIVACY_CONSENT=Y' >> /root/.bashrc && \
    echo 'export RMW_IMPLEMENTATION=rmw_fastrtps_cpp' >> /root/.bashrc && \
    echo 'export AMENT_PREFIX_PATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy:$AMENT_PREFIX_PATH' >> /root/.bashrc && \
    echo 'export PYTHONPATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/rclpy:$PYTHONPATH' >> /root/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/rclpy/:/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib/:$LD_LIBRARY_PATH' >> /root/.bashrc

# Set standard ENV as well for non-interactive shells
ENV ROS_DISTRO=jazzy \
    AMENT_PREFIX_PATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy \
    PYTHONPATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/rclpy \
    LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/rclpy/:/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib/

RUN printf '#!/bin/bash\nexec /isaac-sim/python.sh "$@"\n' > /usr/local/bin/python3 && chmod +x /usr/local/bin/python3
RUN printf '#!/bin/bash\nsource /opt/isaac_bridge_msgs/local_setup.bash > /dev/null 2>&1\nexec python3 "$@"\n' > /usr/local/bin/python3-arena && chmod +x /usr/local/bin/python3-arena

ENTRYPOINT [ "bash" ]