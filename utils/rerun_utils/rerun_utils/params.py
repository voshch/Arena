"""Node parameters of the rerun bridge."""

from __future__ import annotations

from arena_rclpy_mixins.params import Param, Range

WEB_PORT = Param[int]('web_port', 9090, 'Port of the rerun web viewer.', constraint=Range(lo=1, hi=65535))
GRPC_PORT = Param[int]('grpc_port', 9876, 'Port of the rerun gRPC server.', constraint=Range(lo=1, hi=65535))
BRIDGE_PARAMS = (WEB_PORT, GRPC_PORT)
