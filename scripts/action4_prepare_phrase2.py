"""Thin published-phrase2 admission entry; no generation or source-data writes."""
import argparse,json
from data.action4_paths import PROJECT_ROOT,contained_path
from data.action4_phrase2_admission import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=12)
    a=p.parse_args()
    config=json.loads(contained_path(PROJECT_ROOT,a.config,exists=True).read_text())
    if config.get('dataset_root')!='data/phrase2' or config.get('dataset_release')!='phrase2_6500_v001':
        raise ValueError('Explicit published6500 consumer required')
    if config['manifest_paths']!={s:f'data/phrase2/{s}/annotations.jsonl' for s in ('train','eval','test')}:
        raise ValueError('Consumer split roots differ')
    if not 1<=a.workers<=72:raise ValueError('CPU budget exceeded')
    prepare(a.output,a.workers)
