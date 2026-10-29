"""Receipt-bound formal128 inspection; no repeated geometry or PNG hashing."""
from pathlib import Path
from copy import deepcopy
import argparse,collections,concurrent.futures,hashlib,json,math,os,multiprocessing
import numpy as np
import torch
from PIL import Image,ImageDraw
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data import action4_coverage1000_queue as queue
from data.action4_coverage1000_visual_pack import annotate,from_tensor
from data.action4_pose_rgb_batch import prepare_pose_rgb
from data.action4_dataset import prepare_full_images
from data.action4_phrase2_visibility import canonical_visibility
from data.action4_occlusion500_similarity import fingerprint,write

OUT=R/'outputs/coverage1000_v001'

def pin_exporter(counter):
    with counter.get_lock():
        index=counter.value;counter.value+=1
    os.sched_setaffinity(0,{4+index});torch.set_num_threads(1)

def export_one(item):
    torch.set_num_threads(1)
    slot,pointer,receipt,root,*scope=item
    payload=json.loads(contained_path(R,pointer,exists=True).read_text())
    if payload['slot']!=slot:raise ValueError('Preview packet/slot differs')
    r=deepcopy(payload['record'])
    if r['sample_id']!=slot['sample_id'] or r['split']!=slot['split']:raise ValueError('Inspection sample/split differs')
    if not scope and r['split']!='train':raise ValueError('Formal128 preview must use train identities')
    if scope and (scope!=['final_accepted'] or r.get('status')!='training_accepted'):raise ValueError('Final inspection requires an accepted batch record')
    # Match the already defined training visibility conversion, without moving
    # any point, relabeling IDs, or publishing the still-unreviewed candidate.
    for obj in r['instances']:
        g=obj['canonical8_geometry'];flags=canonical_visibility(np.array(r['corners_3d_mm'])/1000,np.array(g['object_to_capture_cv_m']),g['ray_clear_to_virtual_point'],g['point_inframe'])
        obj['corner_visible']=flags['visible']
    target=Path(root)/slot['sample_id'];target.mkdir(exist_ok=False)
    with Image.open(contained_path(R,r['image'],exists=True)) as im:native=im.convert('RGB')
    overlay=annotate(native,r);overlay.save(target/'native_overlay.png')
    x=torch.from_numpy(np.array(native)).permute(2,0,1).contiguous()
    b0,affines=prepare_full_images([x],size=(1024,768),device='cpu');b0image=from_tensor(b0[0]);b0image.save(target/'B0_actual_input.png')
    items=[{'image':x,'record':r,'instance':i} for i in range(2)]
    a0,geometry,metadata=prepare_pose_rgb(items,device='cpu',jitter_config={'center_fraction':.15,'scale_range':[.65,1.35]})
    masks=[];nativepatches=[];b0patches=[];a0patches=[];inputstats=[]
    for i,obj in enumerate(r['instances']):
        box=np.array(obj['bbox_full_xyxy']);bounds=(max(0,math.floor(box[0]-12)),max(0,math.floor(box[1]-12)),min(r['width'],math.ceil(box[2]+12)),min(r['height'],math.ceil(box[3]+12)))
        raw=native.crop(bounds);raw.save(target/f'i{i+1}_native_crop.png')
        patch=overlay.crop(bounds);patch.save(target/f'i{i+1}_native_overlay.png');nativepatches.append(patch)
        with Image.open(contained_path(R,obj['mask_visible'],exists=True)) as im:v=np.asarray(im.crop(bounds))>0
        W,H=r['width'],r['height'];expanded=tuple(np.array(bounds)+[W,H,W,H])
        with Image.open(contained_path(R,obj['mask_amodal_expanded'],exists=True)) as im:a=np.asarray(im.crop(expanded))>0
        colors=np.zeros((*v.shape,3),dtype=np.uint8);colors[a]=(255,60,60);colors[v]=(20,220,20);mask=Image.fromarray(colors);mask.save(target/f'i{i+1}_visible_amodal.png');masks.append(mask)
        roi=from_tensor(a0[i]);roi.save(target/f'A0_i{i+1}_actual_roi.png');a0patches.append(roi)
        b=DataAdapter.transform_points(box.reshape(2,2),affines[0]).reshape(-1);bounds_b=(max(0,math.floor(b[0]-3)),max(0,math.floor(b[1]-3)),min(1024,math.ceil(b[2]+3)),min(768,math.ceil(b[3]+3)))
        patch=b0image.crop(bounds_b);patch.save(target/f'B0_i{i+1}_actual_pixels.png');b0patches.append(patch)
        inputstats.append({'instance_id':i+1,'native_bbox_wh':(box[2:]-box[:2]).tolist(),'B0_bbox_wh':(b[2:]-b[:2]).tolist(),
            'A0_shape':list(a0[i].shape),'A0_corner_visible':geometry['visible'][i].tolist(),'A0_corners_hm':geometry['corners'][i].tolist(),
            'sparse_surface_points':receipt['surface_validation'][i]['points']})
    tile=Image.new('RGB',(1700,302),'#f4f4f4');draw=ImageDraw.Draw(tile);m=receipt['stats']
    desc=f"{slot['sample_id']} / {slot['group']}  widths={m['widths'][0]:.3f},{m['widths'][1]:.3f}  R={m['ratio']:.2f} edge={m['d_edge']:.3f} center={m['center_distance']:.3f} o={r['instances'][0]['occlusion_inframe']:.2f},{r['instances'][1]['occlusion_inframe']:.2f}"
    draw.text((4,2),desc,fill='black')
    full=overlay.copy();full.thumbnail((252,250));tile.paste(full,(0,30))
    for i in range(2):
        patch=nativepatches[i].copy();patch.thumbnail((252,250));tile.paste(patch,(256+i*256,30))
        patch=a0patches[i].resize((224,224));tile.paste(patch,(772+i*228,40))
        patch=b0patches[i].copy();patch.thumbnail((116,140));tile.paste(patch,(1232+i*120,30))
        patch=masks[i].copy();patch.thumbnail((108,220));tile.paste(patch,(1474+i*112,30))
    draw.text((5,280),'Full image + full-box/ID overlay',fill='black');draw.text((260,280),'Native body crops (never upscaled)',fill='black');draw.text((775,280),'Actual A0 448 ROI, displayed224',fill='black');draw.text((1230,280),'Actual B0 body pixels',fill='black');draw.text((1474,280),'V/A masks',fill='black')
    tile.save(target/'ROW.jpg',quality=95)
    result={'slot':slot,'sample_id':r['sample_id'],'candidate':pointer,'image':r['image'],'rgb_sha256':r['rgb_sha256'],
        'row_image':str((target/'ROW.jpg').relative_to(R)),'inspection_root':str(target.relative_to(R)),'actual_metrics':m,'input_stats':inputstats,
        'validation_reused_from':'SQLite preview_checked host receipt; original packet/frame immutable','surface_validation':receipt['surface_validation']}
    (target/'INPUT_RECEIPT.json').write_text(json.dumps(result));return result

def build(workers=16):
    if not 1<=workers<=16:raise ValueError('Preview exporter CPU budget')
    cfg=json.loads((R/'configs/data/action4_coverage1000_v001.json').read_text());plan=json.loads((R/cfg['plan_path']).read_text())
    if queue.counts().get('preview_checked')!=128:raise ValueError('Wait for all128 host-checked previews')
    done=json.loads((OUT/'runs/preview_v001/COMPLETE.json').read_text())
    if done['status']!='AWAIT_ROOT_FORMAL128_VISUAL_REVIEW':raise ValueError('Render stage must have ended')
    root=OUT/'formal128_review_v001';root.mkdir(exist_ok=False);db=queue.connect()
    rows={r['slot']:r for r in db.execute("SELECT * FROM slots WHERE status='preview_checked'")};db.close()
    if set(rows)!=set(plan['preview_slot_ids']):raise ValueError('Preview identities changed')
    selected=[];raw=[]
    for slot_id in plan['preview_slot_ids']:
        row=rows[slot_id];slot=json.loads(row['spec']);receipt=json.loads(row['record']);pointer=receipt['candidate']
        selected.append((slot,pointer,receipt,str(root)));raw.append(json.loads((R/pointer).read_text())['record'])
    counter=multiprocessing.Value('i',0)
    with concurrent.futures.ProcessPoolExecutor(workers,initializer=pin_exporter,initargs=(counter,)) as pool:results=list(pool.map(export_one,selected,chunksize=1))
    sheets=[]
    for page in range(16):
        canvas=Image.new('RGB',(1700,302*8),'white')
        for i,row in enumerate(results[page*8:(page+1)*8]):
            with Image.open(R/row['row_image']) as im:canvas.paste(im,(0,i*302))
        path=root/f'REVIEW_{page:02d}.jpg';canvas.save(path,quality=95);sheets.append(str(path.relative_to(R)))
    report={'status':'HOST128_PASSED_AWAIT_ROOT_VISUAL_REVIEW','images':128,'targets':256,'fixed_id_corners':2048,'all_train':True,
        'protocol_id':cfg['protocol_id'],'preview_image_identity':fingerprint([dict(r,attempt=json.loads((R/row['candidate']).read_text())['proposal']['phrase2']['attempt']) for r,row in zip(raw,results)]),
        'sheets':sheets,'rows':results,'formal_dataset_released':False,'independent_pixel_host_receipts_reused':True}
    write(root/'PREVIEW_REVIEW_BUNDLE.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['rows','sheets']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=16);a=p.parse_args();build(a.workers)
