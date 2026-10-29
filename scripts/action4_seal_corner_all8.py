"""Thin isolated entry for completed corner All8 research sealing."""
import importlib.util
from pathlib import Path
root=Path('/data/zy/vln/a-test')
p=root/'src/eval/action4_corner_research_seal.py'
spec=importlib.util.spec_from_file_location('_corner_research_sealer',p)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
if __name__=='__main__':module.main()
