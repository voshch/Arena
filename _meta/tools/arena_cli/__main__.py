"""Bootstrap for the arena CLI, run by the shell shim."""

import sys

if sys.version_info < (3, 10):
    sys.stderr.write("arena: python >= 3.10 required\n")
    sys.exit(1)

import cli

cli.main(prog_name="arena")
