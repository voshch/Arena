from __future__ import annotations

import subprocess
import sys
import time
import uuid

import pytest

SERVER = """
import os
import sys
import threading
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.signals import SignalHandlerOptions
from std_srvs.srv import Trigger
rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
node = rclpy.create_node("supervisor_discovery_peer")
node.create_service(Trigger, sys.argv[1], lambda request, response: response)
threading.Thread(target=lambda: (sys.stdin.read(), rclpy.shutdown()), daemon=True).start()
print("READY", flush=True)
os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
try:
    rclpy.spin(node)
except ExternalShutdownException:
    pass
"""


def test_supervisor_discovers_a_service_another_process_serves_and_gives_up_on_a_missing_one(rclpy_context):
    if rclpy_context is None:
        pytest.skip("ROS2 not available")
    from arena_bringup.supervisor import Supervisor

    service = f"/supervisor_probe_{uuid.uuid4().hex[:8]}/register_env"
    peer = subprocess.Popen([sys.executable, "-c", SERVER, service], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    node = rclpy_context.create_node(f"supervisor_probe_{uuid.uuid4().hex[:8]}")
    try:
        assert peer.stdout is not None
        assert "READY" in (line.strip() for line in peer.stdout)
        deadline = time.monotonic() + 30.0
        while "supervisor_discovery_peer" not in node.get_node_names():
            assert time.monotonic() < deadline, "the peer node never joined the graph"
            time.sleep(0.1)
        sup = Supervisor(node)
        assert sup.discovers(service, 20.0, 30.0)
        started = time.monotonic()
        assert not sup.discovers(f"{service}_missing", 0.5, 15.0)
        assert time.monotonic() - started < 15.0
    finally:
        node.destroy_node()
        assert peer.stdin is not None
        peer.stdin.close()
        peer.wait()
