"""Publish a fully reviewed fixed-template dataset through explicit release refs."""
import argparse,json,sys
from pathlib import Path
ROOT=Path('/data/zy/vln/a-test');sys.path.insert(0,str(ROOT/'src'))
def main():
 from data.action4_template_assembly import References,assemble
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--ledger',required=True);parser.add_argument('--release',required=True);args=parser.parse_args()
 reader=References();ledger=reader.path(args.ledger);release=reader.path(args.release)
 result=assemble({'path':str(ledger.relative_to(ROOT)),'sha256':reader.sha(ledger)},{'path':str(release.relative_to(ROOT)),'sha256':reader.sha(release)})
 print(json.dumps({'status':'DATASET_PUBLISHED','count':result['commit']['count'],'output_layout':result['commit']['output_layout'],'publication':result['publication']},indent=2))
if __name__=='__main__':main()
