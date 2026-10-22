#!/usr/bin/env python3
"""Thin launcher for the full-size component validation harness."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from check_action4_model_core import main
if __name__=='__main__': main()
