import rclpy
from arena_bringup.constants import SimSimulator
from arena_rclpy_mixins.params import Param, ParamGroup


class RuntimeParams(ParamGroup):
    SIM = Param[SimSimulator]('sim', SimSimulator.DUMMY, 'Simulator backend the runtime drives.')
    PHYSICS_DT = Param[float]('physics_dt', 0.0333, 'Physics step of the simulator in seconds.')
    HEARTBEAT_PERIOD = Param[float]('heartbeat_period_sec', 1.0, 'Seconds between sweeps of the env heartbeats.')
    HEARTBEAT_TIMEOUT = Param[float]('heartbeat_timeout_sec', 30.0, 'Seconds without a heartbeat before an env is evicted.')
    RESET_HOLD_TIMEOUT = Param[float]('reset_hold_timeout_sec', 30.0, 'Seconds without a heartbeat an env may take while it holds the reset before it is evicted.')
    BOOTSTRAP_TIMEOUT = Param[float]('bootstrap_timeout_sec', 0.0, 'Seconds a reserved env slot may wait for its first heartbeat before it is evicted (0 = never).')
    SLOT_BUFFER = Param[float]('slot_buffer', 5.0, 'Margin in metres around the bounding box of an env when packing its world slot.')
    ENV_N = Param[int]('env_n', 0, 'Number of envs the runtime spawns at startup.')
    ENV_ARGS = Param[list[str]]('env_args', [], 'Launch arguments as key:=value strings every spawned env gets.', type_=rclpy.Parameter.Type.STRING_ARRAY)
    LOCKSTEP_CHANNELS = Param[list[str]]('lockstep.channels', [], 'Lockstep channel specs name|topic|type|period_s|hard registered at startup.', type_=rclpy.Parameter.Type.STRING_ARRAY)
    LOCKSTEP_AUTOSTART = Param[bool]('lockstep.autostart', False, 'true = start the lockstep scheduler at startup.')
    LOCKSTEP_TARGET_RTF = Param[float]('lockstep.target_rtf', 0.0, 'Real-time factor the autostarted lockstep aims for (0 = as fast as the gates allow).')
    LOCKSTEP_PAUSED = Param[bool]('lockstep.paused', True, 'true = the autostarted lockstep begins paused.')
