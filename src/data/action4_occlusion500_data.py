"""Read the isolated additional batch through the existing A0/B0 item pipeline."""
import json
import numpy as np
import torch
from PIL import Image
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_stage2_data import Stage2Units

class BatchImages:
    def __init__(self,split,role):
        root=contained_path(R,'data/phrase2/.staging_batches/occlusion500_v001',exists=True)
        self.records=[json.loads(line) for line in (root/split/'annotations.jsonl').read_text().splitlines()]
        assert len(self.records)=={'train':400,'eval':50,'test':50}[split]
        assert len({r['sample_id'] for r in self.records})==len(self.records)
        adapter=DataAdapter(R)
        for r in self.records:
            assert r['split']==split and r['batch_id']=='occlusion500_v001' and r['dataset_root']=='data/phrase2'
            assert [o['instance_id'] for o in r['instances']]==[1,2]
            assert contained_path(R,r['image'],exists=True).is_relative_to(root/split)
            adapter.validate_sample(r,check_image=False)
        self.units=[(i,j) for i in range(len(self.records)) for j in range(2)] if role=='A0' else [(i,None) for i in range(len(self.records))]
    def __len__(self):return len(self.units)
    def __getitem__(self,index):
        i,j=self.units[index];r=self.records[i]
        with Image.open(contained_path(R,r['image'],exists=True)) as im:rgb=torch.from_numpy(np.array(im.convert('RGB'))).permute(2,0,1)
        return {'image':rgb,'record':r,'instance':j}

class AdditionalBatchUnits(Stage2Units):
    """Override selection only; the existing ROI/mask/XYZ/visibility processing runs."""
    def __init__(self,cfg,split='train'):
        assert cfg['mode']=='integration' and cfg['allow_unpublished_interface_validation']
        self.cfg=cfg;self.split=split;self.data=BatchImages(split,cfg['role']);self.indices=list(range(len(self.data)))
        if cfg['role']=='A0':
            expected=f'data/phrase2/.staging_batches/occlusion500_v001/{split}/geometry'
            assert cfg['surface_source']['label_roots'][split]==expected

def surface_source():
    return {'admission':'outputs/phrase2_occlusion500_v001/INTERFACE_GEOMETRY_ADMISSION.json','regions':'data/action4_surface_regions_v001/centers.json',
            'label_roots':{s:f'data/phrase2/.staging_batches/occlusion500_v001/{s}/geometry' for s in ['train','eval','test']}}
