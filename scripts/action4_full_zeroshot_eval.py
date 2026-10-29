"""Thin entry for the full-development frozen YOLO evaluation runner."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval.action4_full_zeroshot_eval import main

if __name__ == "__main__":
    main()
