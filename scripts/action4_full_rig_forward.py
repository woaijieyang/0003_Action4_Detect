#!/usr/bin/env python3
"""Thin entry for diagnostic FK snapshot/forward operations."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from data.action4_full_rig_fit import main
if __name__=="__main__":main()
