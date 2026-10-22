"""Public entry to explicit Action4 persistent-scene execution."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from data.action4_fast_launcher import main
if __name__=="__main__":
    raise SystemExit(main())
