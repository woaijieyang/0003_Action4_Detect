"""Prepare existing stage4 trainer inputs; does not launch training."""
import argparse
from model.training.action4_stage1_prepare import prepare_corner

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--run-id",required=True)
    args=parser.parse_args()
    print(prepare_corner(args.config,args.run_id))
