from arena_rclpy_mixins.params import Param, ParamGroup


class NodeParams(ParamGroup):
    OPTIM = Param[str]('optim', '', 'Optimization profile name handed to the robot loaders (empty = none).')
    PREFIX = Param[str]('prefix', '', 'Entity name prefix of the env (empty = none).')
    ROBOT_TRAIN = Param[bool]('robot.train', False, 'true = a training agent publishes cmd_vel directly and the mobile adapter skips its own navigation stack.')
    ROBOT = Param[str]('robot', '', 'Fleet robot specs the env spawns at startup.')
