"""Compatibility entry point for the adaptive-attack generator."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_adaptive import main


if __name__ == "__main__":
    main()
