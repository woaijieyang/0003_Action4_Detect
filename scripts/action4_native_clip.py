"""Thin entry for a registered native RGB tracking diagnostic."""
import argparse
from eval.action4_native_clip import run_clip
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True)
    run_clip(p.parse_args().output)
