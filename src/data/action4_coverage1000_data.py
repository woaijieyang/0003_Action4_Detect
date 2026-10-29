"""Isolated batch integration through the existing, unchanged A0/B0 item code."""
import json
import numpy as np
import torch
from PIL import Image
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_stage2_data import Stage2Units

class CoverageImages:
    def __init__(self,split,role):
        cfg=json.loads((R/'configs/data/action4_coverage1000_v001.json').read_text());root=contained_path(R,cfg['candidate_root'],exists=True)
        self.records=[json.loads(line) for line in (root/split/'annotations.jsonl').read_text().splitlines()]
        if len(self.records)!=cfg['split_counts'][split] or len({r['sample_id'] for r in self.records})!=len(self.records):raise ValueError('Incomplete or duplicate coverage split')
        for row in self.records:
            if row['split']!=split or row['batch_id']!=cfg['batch_id'] or row['status']!='training_accepted' or [o['instance_id'] for o in row['instances']]!=[1,2]:raise ValueError('Wrong batch identity')
            if not contained_path(R,row['image'],exists=True).is_relative_to(root/split):raise ValueError('Candidate image escaped its explicit split')
            DataAdapter(R).validate_sample(row,check_image=False)
        self.units=[(i,j) for i in range(len(self.records)) for j in range(2)] if role=='A0' else [(i,None) for i in range(len(self.records))]
    def __len__(self):return len(self.units)
    def __getitem__(self,index):
        i,j=self.units[index];row=self.records[i]
        with Image.open(contained_path(R,row['image'],exists=True)) as im:rgb=torch.from_numpy(np.array(im.convert('RGB'))).permute(2,0,1)
        return {'image':rgb,'record':row,'instance':j}

class CoverageBatchUnits(Stage2Units):
    def __init__(self,cfg,split='train'):
        if cfg['mode']!='integration' or not cfg.get('allow_unpublished_interface_validation'):raise ValueError('Unpublished batch is integration-only')
        self.cfg=cfg;self.split=split;self.data=CoverageImages(split,cfg['role']);self.indices=list(range(len(self.data)))
        if cfg['role']=='A0' and cfg['surface_source']['label_roots'][split]!=f'data/phrase2/.staging_batches/coverage1000_v001/{split}/geometry':raise ValueError('No fallback to old geometry')

def surface_source():
    return {'admission':'outputs/coverage1000_v001/INTERFACE_GEOMETRY_ADMISSION.json','regions':'data/action4_surface_regions_v001/centers.json',
        'label_roots':{s:f'data/phrase2/.staging_batches/coverage1000_v001/{s}/geometry' for s in ['train','eval','test']}}
