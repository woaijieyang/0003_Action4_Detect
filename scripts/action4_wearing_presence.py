from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':runpy.run_path(str(ROOT/'outputs/action4_wearing_presence_actual_v001/runner.py'),run_name='__main__')
