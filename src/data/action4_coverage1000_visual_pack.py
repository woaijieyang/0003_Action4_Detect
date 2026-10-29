"""Host validation and actual A0/B0 input exports for diagnostic or preview RGB."""
from pathlib import Path
import json,sys,os,math
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data import DataAdapter
from data.action4_coverage1000_quality import validate
from data.action4_phrase2_host_worker import surface_check
from data.action4_pose_rgb_batch import prepare_pose_rgb
from data.action4_dataset import prepare_full_images
from data.action4_phrase2_visibility import canonical_visibility

EDGES=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def from_tensor(x):
    mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
    return Image.fromarray(((x.cpu()*std+mean).clamp(0,1)*255).round().byte().permute(1,2,0).numpy())

def annotate(image,record,affine=None):
    result=image.copy();draw=ImageDraw.Draw(result);A=np.eye(3) if affine is None else np.array(affine)
    for obj in record['instances']:
        uv=DataAdapter.transform_points(obj['corners_px'],A)
        box=DataAdapter.transform_points(np.array(obj['bbox_full_xyxy']).reshape(2,2),A)
        draw.rectangle(tuple(box.reshape(-1)),outline=(20,255,20),width=2)
        g=obj['canonical8_geometry'];flags=canonical_visibility(np.array(record['corners_3d_mm'])/1000,np.array(g['object_to_capture_cv_m']),g['ray_clear_to_virtual_point'],g['point_inframe'])
        for a,b in EDGES:draw.line([tuple(uv[a]),tuple(uv[b])],fill=(255,50,50),width=2)
        for k,point in enumerate(uv):
            x,y=point;draw.ellipse((x-3,y-3,x+3,y+3),outline=(255,255,0),fill=(255,255,0) if flags['visible'][k] else None,width=1)
            draw.text((x+4,y+2),str(k),fill=(255,255,0),stroke_width=1,stroke_fill=(0,0,0))
    return result

def run(root):
    torch.set_num_threads(1);root=contained_path(R,root,exists=True)
    cfgpath='configs/data/action4_coverage1000_arm_pose_light_v001.json' if root.name=='visual_probe_pose_light' else 'configs/data/action4_coverage1000_feasibility_arm_v001.json'
    cfg=json.loads((R/cfgpath).read_text())
    bcfg=json.loads((R/'configs/model/action4_b0_photometric_v001.json').read_text());fullsize=tuple(bcfg['preprocess']['full_size'])
    rows=[]
    for pointer in sorted(root.glob('*/COMPLETE.json')):
        receipt=json.loads(pointer.read_text())
        for case in receipt['receipts']:
            r=json.loads(contained_path(R,case['record'],exists=True).read_text());snapshot=json.loads(contained_path(R,case['snapshot'],exists=True).read_text())
            slot=snapshot['slot'];p=snapshot['proposal'];ok,reason,metrics=validate(r,p,slot,cfg);surface=surface_check(r)
            directory=contained_path(R,case['record'],exists=True).parent/'inspection';directory.mkdir(exist_ok=False)
            with Image.open(contained_path(R,r['image'],exists=True)) as im:native=im.convert('RGB')
            overlay=annotate(native,r);overlay.save(directory/'native_overlay.png')
            rgb=torch.from_numpy(np.array(native)).permute(2,0,1).contiguous()
            full,aff=prepare_full_images([rgb],size=fullsize,device='cpu');b0=from_tensor(full[0]);b0.save(directory/'B0_actual_input.png')
            annotate(b0,r,aff[0]).save(directory/'B0_actual_overlay.png')
            items=[{'image':rgb,'record':r,'instance':i} for i in range(2)]
            crops,_,metadata=prepare_pose_rgb(items,device='cpu',jitter_config={'center_fraction':.15,'scale_range':[.65,1.35]})
            details=[]
            for i,obj in enumerate(r['instances']):
                from_tensor(crops[i]).save(directory/f'A0_i{i+1}_actual_roi.png')
                box=np.array(obj['bbox_full_xyxy']);pad=16;bounds=(max(0,math.floor(box[0]-pad)),max(0,math.floor(box[1]-pad)),min(r['width'],math.ceil(box[2]+pad)),min(r['height'],math.ceil(box[3]+pad)))
                native.crop(bounds).save(directory/f'i{i+1}_native_crop.png');overlay.crop(bounds).save(directory/f'i{i+1}_native_overlay.png')
                with Image.open(contained_path(R,obj['mask_visible'],exists=True)) as im:visible=np.asarray(im)>0
                with Image.open(contained_path(R,obj['mask_amodal_expanded'],exists=True)) as im:amodal=np.asarray(im)>0
                W,H=r['width'],r['height'];amodal=amodal[H:2*H,W:2*W]
                mask=np.zeros((H,W,3),dtype=np.uint8);mask[amodal]=(255,60,60);mask[visible]=(20,220,20)
                Image.fromarray(mask).crop(bounds).save(directory/f'i{i+1}_visible_amodal_crop.png')
                details.append({'instance_id':i+1,'occlusion_inframe':obj['occlusion_inframe'],'native_bbox_wh':(box[2:]-box[:2]).tolist(),
                    'B0_bbox_wh':((box[2:]-box[:2])*np.diag(aff[0])[:2]).tolist(),'visible_B0_pixels':obj['visible_area_at_fullinput_px'],'A0_meta':metadata[i]})
            thumb=native.resize((650,493));thumb_overlay=overlay.resize((650,493));panel=Image.new('RGB',(1300,1008),(245,245,245));panel.paste(thumb,(0,22));panel.paste(thumb_overlay,(650,22))
            draw=ImageDraw.Draw(panel);draw.text((8,5),f"DIAGNOSTIC arm asset / {slot['group']} / {r['sample_id']} / pixel quota={ok}",fill='black')
            for i in range(2):
                patch=Image.open(directory/f'i{i+1}_native_overlay.png').convert('RGB');patch.thumbnail((300,260));panel.paste(patch,(i*325,545));draw.text((i*325+8,525),f"native crop i{i+1}; no upscaling",fill='black')
                patch=Image.open(directory/f'A0_i{i+1}_actual_roi.png').convert('RGB').resize((250,250));panel.paste(patch,(650+i*325,545));draw.text((650+i*325+8,525),f"A0 actual448 ROI i{i+1}",fill='black')
                patch=Image.open(directory/f'i{i+1}_visible_amodal_crop.png').convert('RGB');patch.thumbnail((300,180));panel.paste(patch,(i*325,820))
            b0small=b0.copy();b0small.thumbnail((600,190));panel.paste(b0small,(650,815));panel.save(directory/'REVIEW_PANEL.jpg',quality=95)
            rows.append({'sample_id':r['sample_id'],'slot':slot,'image':r['image'],'actual_quota_pass':ok,'reason':reason,'metrics':metrics,'surface':surface,
                'training_inputs':details,'review_panel':str((directory/'REVIEW_PANEL.jpg').relative_to(R)),'inspection_dir':str(directory.relative_to(R))})
    result={'status':'HOST_PIXEL_SURFACE_AND_INPUTS_CHECKED_VISUAL_REVIEW_REQUIRED','rows':rows,'formal_dataset_published':False,'B0_full_size':fullsize,'A0_roi':448}
    (root/'HOST_VALIDATION.json').write_text(json.dumps(result,indent=2));print(json.dumps({'rows':len(rows),'actual_quota_pass':sum(r['actual_quota_pass'] for r in rows),'B0_size':fullsize,'A0':448}))

if __name__=='__main__':run(sys.argv[1])
