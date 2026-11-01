#! /usr/bin/env python3
from arena_runtime.env.episode_node import EpisodeNode


def main(args: list[str] | None = None) -> None:
    del args
    EpisodeNode.run_main(aiomonitor=True)


if __name__ == '__main__':
    import time

    time.sleep(5)
    main()
