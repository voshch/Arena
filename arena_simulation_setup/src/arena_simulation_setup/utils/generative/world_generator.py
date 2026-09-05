import sys
from pathlib import Path

from arena_simulation_setup.tree.World import WorldDescription, WorldIdentifier

from . import WorldGenerator, WorldGeneratorType

__all__ = ['WorldGenerator', 'WorldGeneratorType']


def test_generate(out: str, name: str, config: dict) -> Path:
    gen = WorldGenerator(WorldGeneratorType(name), config)
    return WorldIdentifier(out).resolve_write_sync().save(WorldDescription.from_levels(gen.compute()))


def main(argv: list[str] = sys.argv) -> None:
    import json
    import os

    if len(argv) == 3:
        result = test_generate(argv[1], argv[2], {})
        print(f'Generated world saved to {result}')
    elif len(argv) == 4:
        result = test_generate(argv[1], argv[2], json.loads(argv[3]))
        print(f'Generated world saved to {result}')
    else:
        print(f'usage: {os.path.basename(__file__)} <world_name> <generator> [<config>]')
        sys.exit(1)


if __name__ == '__main__':
    main()
