"""Real-frame paired geometry review. Assumed K, demo reference only."""
import json
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from data.action4_paths import PROJECT_ROOT as R, contained_path
from eval.action4_geometry_pipeline import GeometryPipeline
from eval.action4_metrics import bbox_iou
from eval.action4_video import draw_cuboids
from model.training.action4_runtime import atomic_json


def run(output):
    torch.set_num_threads(2)
    out=contained_path(R, R/output, exists=True)
    source=R/'outputs/action4_current_geometry_comparison_v001'
    pipeline=GeometryPipeline(source/'B0.pt',source/'A0.pt',source/'residual.pt','cuda:0')
    context=json.loads((R/'outputs/action4_execution_source_release_v006/source_context.json').read_text())
    K=np.array(context['K_native'])
    record=json.loads((R/'data/eval/annotations.jsonl').read_text().splitlines()[0])
    canonical=np.array(record['corners_3d_mm'])/1000
    manifest=json.loads((R/'data/action4_real/cache_v001/frames_manifest.json').read_text())
    references=json.loads((R/'outputs/action4_demo_alignment_v001/REFERENCE_OVERLAY_EXTRACTION.json').read_text())['frames']
    rows=[];panels=[]
    for ref in references:
        index=ref['frame_id']
        path=contained_path(R, R/manifest['frames'][index]['image'],exists=True)
        with Image.open(path) as im:rgb=im.convert('RGB')
        assert rgb.size==(3248,2464), 'Assumed intrinsics size mismatch'
        tensor=torch.from_numpy(np.array(rgb)).permute(2,0,1)
        variants=pipeline.predict(tensor,K,canonical)
        scale=np.array([950/rgb.width,720/rgb.height]);preds=variants['raw']
        matrix=np.array([[bbox_iou((np.array(p['bbox_full_xyxy']).reshape(2,2)*scale).reshape(4),g['bbox_demo_px']) for g in ref['reference_overlay_components']] for p in preds])
        pairs=[]
        if len(preds) and matrix.size:
            x,y=linear_sum_assignment(-matrix);pairs=[float(matrix[u,v]) for u,v in zip(x,y)]
        row={'frame_id':index,'variants':variants,'box_demo_reference_iou':pairs,'both_reference_iou_ge05':len(pairs)==2 and min(pairs)>=.5}
        rows.append(row)
        strip=Image.new('RGB',(950*3,750))
        for column,(name,instances) in enumerate(variants.items()):
            panel=draw_cuboids(rgb.copy(),instances,line_width=6).resize((950,720))
            draw=ImageDraw.Draw(panel)
            for pred in instances:
                box=(np.array(pred['bbox_full_xyxy']).reshape(2,2)*scale).reshape(4).tolist()
                draw.rectangle(box,outline='lime',width=2)
            panel.save(out/f'{name}_{index:08d}.png');strip.paste(panel,(column*950,30))
            ImageDraw.Draw(strip).text((column*950+10,5),f'frame {index}: {name}; assumed K, not calibrated',fill='white')
        panels.append(strip)
        atomic_json(out/'STATUS.json',{'completed_frames':len(rows),'total':len(references)})
    sheet=Image.new('RGB',(2850,750*len(panels)))
    for i,panel in enumerate(panels):sheet.paste(panel,(0,750*i))
    sheet.save(out/'comparison.jpg',quality=92)
    atomic_json(out/'REPORT.json',{'scope':'real demo-reference diagnostic, not GT accuracy','intrinsics_status':'assumed from synthetic source, not measured real calibration','K':K.tolist(),'checkpoint_directory':str(source.relative_to(R)),'both_reference_frames':sum(r['both_reference_iou_ge05'] for r in rows),'frames':rows,'production_accepted':False})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
