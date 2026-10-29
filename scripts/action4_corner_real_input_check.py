import argparse
from eval.action4_corner_real_input_check import run
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",required=True);run(p.parse_args().output)
