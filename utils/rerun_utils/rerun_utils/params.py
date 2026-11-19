"""Node parameters of the rerun bridge."""

from __future__ import annotations

from arena_rclpy_mixins.params import Param, Range

WEB_PORT = Param[int]('web_port', 9090, 'Port of the rerun web viewer.', constraint=Range(lo=1, hi=65535))
GRPC_PORT = Param[int]('grpc_port', 9876, 'Port of the rerun gRPC server.', constraint=Range(lo=1, hi=65535))
MEMORY_LIMIT = Param[str]('memory_limit', '512MiB', 'Most memory the gRPC server buffers for viewers that connect later, in rerun size syntax (512MiB, 2GB, 25%).')
MAX_RATE_HZ = Param[float]('max_rate_hz', 30.0, 'Most updates a second logged per topic and tf frame (0 = uncapped).', constraint=Range(lo=0.0))
BRIDGE_PARAMS = (WEB_PORT, GRPC_PORT, MEMORY_LIMIT, MAX_RATE_HZ)
