#!/usr/bin/env python3
"""Thin entrypoint for bounded actual CUDA full trainer-state resume verification."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests"))
from check_action4_cuda_resume import main
if __name__ == "__main__":
    main()
