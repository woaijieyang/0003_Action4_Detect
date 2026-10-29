"""Thin entry for the completed ROI adaptation development model pair."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from eval.action4_roi_pair_selection import freeze_pair
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("a0-seal","b0-seal","baseline-input-lock","output"):p.add_argument("--"+name,required=True)
    a=p.parse_args();freeze_pair(a.a0_seal,a.b0_seal,a.baseline_input_lock,a.output)
