"""Published label -> detector target -> native box; no model or score tuning."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from data import DataAdapter
from data.action4_dataset import build_detection_targets
from eval.action4_runner import decode_full_boxes
from eval.action4_metrics import is_ignored


def main():
    torch.set_num_threads(2);root=Path.cwd();cfg=json.loads((root/'configs/model/component_trials_v001/b0_giou_quality.json').read_text())
    cfg['decode']={**cfg['decode'],'score_threshold':.3}
    size=tuple(cfg['preprocess']['full_size']);maxerror=0.;targets=images=0;groups={'heavy':[],'other':[]}
    for split in ['train','eval','test']:
        for line in (root/f'data/{split}/annotations.jsonl').read_text().splitlines():
            record=json.loads(line);matrix,_=DataAdapter.letterbox_affine((record['width'],record['height']),size)
            t=build_detection_targets([record],[matrix],full_size=size)
            shape=t['center'].shape;output={'center_logits':torch.full(shape,-20.),'offset':torch.zeros(1,2,*shape[-2:]),'ltrb':torch.zeros(1,4,*shape[-2:])}
            for j in range(2):
                if not t['instance_mask'][0,j]:continue
                index=int(t['indices'][0,j]);y,x=divmod(index,shape[-1]);output['center_logits'][0,0,y,x]=20
                output['offset'][0,:,y,x]=t['offset'][0,j];output['ltrb'][0,:,y,x]=t['ltrb'][0,j]
            pred=decode_full_boxes(output,[matrix],cfg)[0];gt=[x for x in record['instances'] if not is_ignored(x)]
            assert len(pred)==len(gt),(record['sample_id'],len(pred),len(gt))
            if gt:
                cost=np.array([[np.max(np.abs(np.array(p['bbox_full_xyxy'])-g['bbox_full_xyxy'])) for g in gt] for p in pred]);a,b=linear_sum_assignment(cost);maxerror=max(maxerror,float(cost[a,b].max()))
                for g in gt:
                    full=np.array(g['bbox_full_xyxy']);visible=np.array(g['bbox_visible_xyxy']);shift=np.linalg.norm((full[:2]+full[2:]-visible[:2]-visible[2:])/2)/np.linalg.norm(full[2:]-full[:2]);groups['heavy' if g['occlusion_inframe']>=.4 else 'other'].append(float(shift))
            targets+=len(gt);images+=1
    assert maxerror<.01,maxerror
    result={'status':'PASS','scope':'all2000 label encoding/decoding engineering check, not model eval','images':images,'targets':targets,'max_native_box_error_px':maxerror,'visible_vs_full_center_shift_fraction':{k:{'count':len(v),'median':float(np.median(v)),'p95':float(np.percentile(v,95))} for k,v in groups.items()}}
    out=root/'outputs/action4_detection_contract_v001';out.mkdir(exist_ok=True);(out/'REPORT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
