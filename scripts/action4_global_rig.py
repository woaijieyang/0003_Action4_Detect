#!/usr/bin/env python3
"""Thin Blender entry for G1 fixed metric rig initialization."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from data.action4_global_rig import main
if __name__=="__main__":main()
