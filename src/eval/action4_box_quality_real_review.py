"""Frozen box-quality head and Gaussian Soft-NMS on six real reference frames."""
import json
import numpy as np
import torch
from PIL import Image,ImageDraw
from scipy.optimize import linear_sum_assignment
from data.action4_paths import PROJECT_ROOT as R
from data.action4_dataset import prepare_full_images
from eval.action4_component_pipeline import load_trained_component
from eval.action4_box_quality_pool import candidate_union
from model.action4_box_quality import BoxAwareQuality,box_features
from model.action4_component_experiments import quality_decode
from eval.action4_soft_nms import select
from eval.action4_metrics import bbox_iou
from model.training.action4_runtime import atomic_json

def run():
    torch.set_num_threads(1);out=R/'outputs/action4_box_quality_real_review_v001';base,cfg=load_trained_component(R/'outputs/action4_box_quality_candidate_v001/base_B0_best.pt','B0','cuda:0');head=BoxAwareQuality().to('cuda:0');state=torch.load(R/'outputs/action4_box_quality_nms_audit_v001/head.pt',map_location='cpu',weights_only=True);head.load_state_dict(state['head'],strict=True);head.eval();captured={}
    handle=base.decoder.register_forward_hook(lambda module,args,value:captured.update(feature=value.detach()))
    manifest=json.loads((R/'data/action4_real/cache_v001/frames_manifest.json').read_text());refs=json.loads((R/'outputs/action4_demo_alignment_v001/REFERENCE_OVERLAY_EXTRACTION.json').read_text())['frames'];rows=[];sheet=Image.new('RGB',(1900,750*len(refs)))
    with torch.inference_mode():
        for index,ref in enumerate(refs):
            fid=ref['frame_id']
            with Image.open(R/manifest['frames'][fid]['image']) as im:rgb=im.convert('RGB')
            x,affine=prepare_full_images([torch.from_numpy(np.array(rgb)).permute(2,0,1)],size=tuple(cfg['preprocess']['full_size']),device='cuda:0');dense=base(x);candidates,boxes=candidate_union(dense,affine[0],cfg);f,g=box_features(captured['feature'],boxes,tuple(cfg['preprocess']['full_size']));scores=head(f,g).sigmoid().cpu().tolist()
            new=select([dict(c,score=s) for c,s in zip(candidates,scores)],'soft_gaussian50',threshold=.3);old=quality_decode(dense,affine,cfg,cfg['variant'])[0];variants={'original':old,'box_quality_soft_nms':new};row={'frame_id':fid,'variants':{}}
            for column,(name,predictions) in enumerate(variants.items()):
                scale=np.array([950/rgb.width,720/rgb.height]);image=rgb.resize((950,720));draw=ImageDraw.Draw(image)
                for prediction in predictions:
                    box=(np.array(prediction['bbox_full_xyxy']).reshape(2,2)*scale).reshape(4);draw.rectangle(box.tolist(),outline='lime',width=2);draw.text((box[0],box[1]),f'{prediction["score"]:.3f}',fill='yellow')
                matrix=np.array([[bbox_iou((np.array(p['bbox_full_xyxy']).reshape(2,2)*scale).reshape(4),g['bbox_demo_px']) for g in ref['reference_overlay_components']] for p in predictions]);matches=[]
                if matrix.size:
                    a,b=linear_sum_assignment(-matrix);matches=[float(matrix[i,j]) for i,j in zip(a,b)]
                row['variants'][name]={'predictions':predictions,'demo_reference_iou':matches,'both_reference_ge05':len(matches)==2 and min(matches)>=.5}
                image.save(out/f'{name}_{fid:08d}.png');sheet.paste(image,(column*950,index*750+30));ImageDraw.Draw(sheet).text((column*950+8,index*750+6),f'frame{fid}: {name}; demo reference only, not GT',fill='white')
            rows.append(row)
    handle.remove();sheet.save(out/'comparison.jpg',quality=92);atomic_json(out/'REPORT.json',{'scope':'six real demo-reference frames, not formal GT metrics','head_step':state['step'],'rows':rows,'both_reference_frames':{name:sum(r['variants'][name]['both_reference_ge05'] for r in rows) for name in variants},'production_accepted':False})
if __name__=='__main__':run()
