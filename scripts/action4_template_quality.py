"""Thin final host/GPU3 quality entry for completed fixed-template candidates."""
import sys
from pathlib import Path
ROOT = Path('/data/zy/vln/a-test')
sys.path.insert(0, str(ROOT/'src'))
if __name__ == '__main__':
    from eval.action4_template_quality import main
    raise SystemExit(main())
