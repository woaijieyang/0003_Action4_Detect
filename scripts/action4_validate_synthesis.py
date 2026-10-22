"""Validate generated Action4 candidates through one bounded parallel pipeline."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from eval.action4_synthesis_validation import main
if __name__=="__main__":
    raise SystemExit(main())
