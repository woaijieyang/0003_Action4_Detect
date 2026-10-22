"""Visualize derived XYZ on original crops; outputs are inspection only."""
import json
import numpy as np
from PIL import Image,ImageDraw
from data.action4_paths import PROJECT_ROOT as R

def run():
    out=R/'outputs/action4_surface_labels_v002';report=json.loads((out/'VALIDATION.json').read_text());assert report['status']=='SERIALIZED_GEOMETRY_PASS_VISUAL_REVIEW_PENDING'
    selected=[next(x for x in report['details'] if x['split']==s) for s in ['train','eval','test']];selected+=sorted(report['details'],key=lambda x:x['miss_fraction'],reverse=True)[:3]
    records={s:{r['sample_id']:r for r in (json.loads(l) for l in (R/f'data/{s}/annotations.jsonl').read_text().splitlines())} for s in ['train','eval','test']}
    mesh=np.load(R/'data/DJI_Action4_construct/r3_v001/mesh_v001.npz')['vertices_mm']/1000;lo=mesh.min(0);span=mesh.max(0)-lo;sheet=Image.new('RGB',(896,480*len(selected)))
    for row,selected_row in enumerate(selected):
        s=selected_row['split'];record=records[s][selected_row['sample_id']];obj=next(o for o in record['instances'] if o['instance_id']==selected_row['instance_id']);bbox=np.array(obj['bbox_full_xyxy']);center=(bbox[:2]+bbox[2:])/2;side=max(bbox[2:]-bbox[:2])*1.5;bounds=np.rint(np.r_[center-side/2,center+side/2]).astype(int)
        with Image.open(R/record['image']) as im:crop=im.convert('RGB').crop(tuple(bounds)).resize((448,448))
        labelled=crop.copy();draw=ImageDraw.Draw(labelled);name=record['sample_id']+'_i'+str(obj['instance_id']);data=np.load(R/f'data/{s}/surface_v002_candidate/{name}.npz');uv=(data['native_pixels']-bounds[:2])*448/(bounds[2:]-bounds[:2]);colors=np.clip((data['xyz_object_m']-lo)/span*255,0,255).astype(int)
        for (x,y),color in zip(uv,colors):draw.ellipse((x-1,y-1,x+1,y+1),fill=tuple(color))
        sheet.paste(crop,(0,row*480+32));sheet.paste(labelled,(448,row*480+32));ImageDraw.Draw(sheet).text((5,row*480+4),f'{s} {name} | RGB / object XYZ colors | miss={selected_row["miss_fraction"]:.3%}',fill='white')
    sheet.save(out/'surface_correspondences.jpg',quality=94);(out/'VISUAL_SELECTION.json').write_text(json.dumps(selected,indent=2))
if __name__=='__main__':run()
