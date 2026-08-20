import json
import sys
import time

import rclpy.qos
import std_srvs.srv
import world_generator_msgs.msg
import world_generator_msgs.srv
from arena_rclpy_mixins.ROSParamServer import ROSParamServer
from arena_rclpy_mixins.ServiceNamespace import ServiceNamespace

from arena_simulation_setup.tree.World.World import WorldDescription, WorldIdentifier

from . import alphabet
from .layout import diagnostics_of
from .schema import declare_config_params
from .world_generator import WorldGenerator, WorldGeneratorType

PREVIEW_PIXEL_BUDGET = 800


class WorldGeneratorROS(WorldGenerator, ROSParamServer, ServiceNamespace):
    def _get_parameters(self) -> tuple[WorldGeneratorType, dict, int]:
        name = WorldGeneratorType(self.get_parameter('generator').value)
        seed = self.get_parameter('seed').value
        prefix = f'algorithm.{name.value}'
        raw = self.get_parameters_by_prefix(prefix)
        config = {leaf: param.value for leaf, param in raw.items()}

        self.get_logger().info(f'world generator: "{name}"')
        self.get_logger().info(f'config: {config}')
        self.get_logger().info(f'seed: {seed}')

        return name, config, seed

    def _cb_generate(self, request: std_srvs.srv.Trigger.Request, response: std_srvs.srv.Trigger.Response) -> std_srvs.srv.Trigger.Response:
        try:
            self.update_generator(*self._get_parameters())
            WorldIdentifier(self.get_parameter('world').value).resolve_sync().save(WorldDescription.from_levels(self.compute()), extra_files=self.files())
            response.success = True
            response.message = json.dumps(self.params())  # episode binding for the panel to apply on queue
        except Exception as e:
            response.success = False
            response.message = repr(e)
            self.get_logger().error(f"Failed to generate world: {repr(e)}")

        return response

    def _cb_generate_world(
        self,
        request: world_generator_msgs.srv.GenerateWorld.Request,
        response: world_generator_msgs.srv.GenerateWorld.Response,
    ) -> world_generator_msgs.srv.GenerateWorld.Response:
        """Preview and generate share one path, so what the panel shows is what a save would write."""
        started = time.perf_counter()
        try:
            name, config, seed = self._get_parameters()
            if request.generator:
                name = WorldGeneratorType(request.generator)
                config = {leaf: param.value for leaf, param in self.get_parameters_by_prefix(f'algorithm.{name.value}').items()}
            if request.config:
                config.update(json.loads(request.config))
            if request.sketch:
                config['sketch'] = request.sketch

            self.update_generator(name, config, seed)
            level = self.compute()
            diagnostics = diagnostics_of(level)

            if not request.preview_only:
                WorldIdentifier(request.world or self.get_parameter('world').value).resolve_sync().save(
                    WorldDescription.from_levels(level), extra_files=self.files()
                )

            resolution = request.resolution or max(0.05, max(diagnostics.extent) / PREVIEW_PIXEL_BUDGET)
            response.png, map_origin = level.render(resolution=resolution)
            response.map_origin = [float(map_origin[0]), float(map_origin[1])]
            response.map_resolution = float(resolution)
            frame = self.frame()
            if frame is not None:
                response.grid_origin = [float(frame.origin[0]), float(frame.origin[1])]
                response.grid_pitch = float(frame.pitch)
                response.grid_size = [frame.rows, frame.cols]
            response.normalized = self.normalize(config.get('sketch', ''))
            response.episode_binding = json.dumps(self.params())
            response.components = diagnostics.components
            response.islands = diagnostics.islands
            response.zones = diagnostics.zones
            response.extent = [float(diagnostics.extent[0]), float(diagnostics.extent[1])]
            response.warnings = [
                world_generator_msgs.msg.SketchWarning(row=note.row, col=note.col, text=note.text) for note in self.warnings
            ]
            response.success = True
        except Exception as error:
            response.success = False
            response.message = repr(error)
            self.get_logger().error(f'preview failed: {error!r}')

        response.compile_ms = (time.perf_counter() - started) * 1e3
        if request.include_alphabet:
            response.alphabet = self._alphabet
        return response

    def __init__(self):
        ROSParamServer.__init__(self, 'world_generator')

        self.declare_parameter('generator', WorldGeneratorType.HALLWAY.value)
        self.declare_parameter('seed', -1)
        self.declare_parameter('world', 'generated')

        for gen_type in WorldGenerator.available():
            model_cls = WorldGenerator.config_model(gen_type)
            declare_config_params(self, f'algorithm.{gen_type.value}', model_cls)

        WorldGenerator.__init__(self, *self._get_parameters())

        self._alphabet = world_generator_msgs.msg.Alphabet(
            entries=[world_generator_msgs.msg.AlphabetEntry(glyph=glyph, arms=list(arms)) for glyph, arms in alphabet.entries()],
            aliases=[
                world_generator_msgs.msg.AlphabetEntry(glyph=char, arms=list(alphabet.arms_of(char) or alphabet.VOID))
                for char in (*alphabet.ASCII_ALIASES, *alphabet.FILL_ALIASES)
            ],
            void_chars=''.join(alphabet.VOID_CHARS),
        )

        self.set_up_services()
        self.get_logger().info('initialized')

    def set_up_services(self):
        self.create_service(std_srvs.srv.Trigger, self.service_namespace('generate_world'), self._cb_generate)
        self.create_service(world_generator_msgs.srv.GenerateWorld, self.service_namespace('generate'), self._cb_generate_world)

        self._alphabet_publisher = self.create_publisher(
            world_generator_msgs.msg.Alphabet,
            self.service_namespace('alphabet'),
            rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._alphabet_publisher.publish(self._alphabet)


def main(argv: list[str] = sys.argv) -> None:
    import os

    import rclpy
    import rclpy.utilities
    from arena_rclpy_mixins.spin import spin_node

    rclpy.init(args=argv)
    argv = rclpy.utilities.remove_ros_args(argv)

    if len(argv) > 1:
        print(f'usage: {os.path.basename(__file__)}')
        sys.exit(1)

    spin_node(WorldGeneratorROS())


if __name__ == '__main__':
    main()
