"""Put the arena_cli dir on sys.path so `import common`/`from features import ...` resolve."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
