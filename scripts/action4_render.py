"""Thin entry for Action 4 conservative mesh preparation and Blender pilot."""

from pathlib import Path
import sys

# Blender's isolated Python does not inherit the project's editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.action4_render import main

if __name__ == "__main__":
    main()
