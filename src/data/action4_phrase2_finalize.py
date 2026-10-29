"""Compatibility CLI; current release requires the global similarity gate."""
from data.action4_phrase2_finalize_v002 import audit,interface_check,release
if __name__=='__main__':
    import argparse,json
    from data.action4_phrase2_similarity import accepted,require_proof
    parser=argparse.ArgumentParser();parser.add_argument('--release',action='store_true');args=parser.parse_args()
    require_proof(accepted());report=audit();interface_check()
    if args.release:release(report)
    print(json.dumps({'audit':report['status'],'interface':'PASS','released':args.release}))
