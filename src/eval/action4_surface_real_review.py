"""Real six-frame new surface A0 review with unchanged original B0."""
import json
import numpy as np
import torch
from PIL import Image,ImageDraw
from data.action4_paths import PROJECT_ROOT as R
from eval.action4_surface_pipeline import SurfacePipeline
from eval.action4_video import draw_cuboids
from eval.action4_metrics import bbox_iou
from model.training.action4_runtime import atomic_json

def run():
    torch.set_num_threads(1);out=R/'outputs/action4_surface_real_review_v001';pipeline=SurfacePipeline(out/'B0.pt',out/'A0.pt',out/'head.pt')
    manifest=json.loads((R/'data/action4_real/cache_v001/frames_manifest.json').read_text());refs=json.loads((R/'outputs/action4_demo_alignment_v001/REFERENCE_OVERLAY_EXTRACTION.json').read_text())['frames'];K=np.array(json.loads((R/'outputs/action4_execution_source_release_v006/source_context.json').read_text())['K_native']);canonical=np.array(json.loads((R/'data/DJI_Action4_construct/r3_v001/asset_manifest.json').read_text())['corners_3d_mm'])/1000
    sheet=Image.new('RGB',(1900,750*len(refs)));rows=[]
    for index,ref in enumerate(refs):
        fid=ref['frame_id']
        with Image.open(R/manifest['frames'][fid]['image']) as im:rgb=im.convert('RGB')
        result=pipeline.predict(torch.from_numpy(np.array(rgb)).permute(2,0,1),K,canonical,seed=20261022+fid*2);result['frame_id']=fid
        for row in result['surface']:
            if row['pose_valid']:
                points=np.array(row['corners_px']);envelope=np.r_[points.min(0),points.max(0)];row['projected_vs_detection_iou']=bbox_iou(envelope,row['bbox_full_xyxy'])
        for column,name in enumerate(['heatmap','surface']):
            valid=[p for p in result[name] if p['corners_px'] is not None];image=draw_cuboids(rgb,valid,line_width=6).resize((950,720));draw=ImageDraw.Draw(image);scale=np.array([950/rgb.width,720/rgb.height])
            for row in result[name]:
                box=(np.array(row['bbox_full_xyxy']).reshape(2,2)*scale).reshape(4);draw.rectangle(box.tolist(),outline='lime',width=2)
                if row['corners_px'] is None:draw.text((box[0],box[1]),'POSE INVALID',fill='yellow')
            image.save(out/f'{name}_{fid:08d}.png');sheet.paste(image,(column*950,index*750+30));ImageDraw.Draw(sheet).text((column*950+8,index*750+6),f'frame{fid} | {name} | frozen B0, assumed K, diagnostic',fill='white')
        rows.append(result);atomic_json(out/'STATUS.json',{'completed':len(rows),'total':len(refs)})
    sheet.save(out/'comparison.jpg',quality=94);atomic_json(out/'REPORT.json',{'scope':'real RGB predicted B0 ROIs; no GT pose/coordinates/masks in inference','head_step':pipeline.head_step,'K_status':'assumed synthetic source K, not real calibration','rows':rows,'valid_poses':sum(p['pose_valid'] for r in rows for p in r['surface']),'detections':sum(len(r['detections']) for r in rows),'production_accepted':False})
if __name__=='__main__':run()
