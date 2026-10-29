"""New-image final inspection selection bound to the completed6500 audit."""
from pathlib import Path
import collections,concurrent.futures,json,math,multiprocessing,os
import numpy as np
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R
from data.action4_coverage1000_similarity import new_records,fingerprint,write,OUT
from data.action4_coverage1000_preview import export_one,pin_exporter

def build():
    audit=json.loads((OUT/'FINAL_AUDIT.json').read_text());proof=json.loads((OUT/'SIMILARITY_COMPLETE.json').read_text());records=new_records()
    if proof['status']!='PASS_GLOBAL_SIMILARITY_AUDIT' or audit['batch_identity']!=fingerprint(records) or audit['global_identity']!=proof['dataset_identity']:raise ValueError('Final audit/similarity identity differs')
    plan=json.loads((OUT/'SLOTS.json').read_text());slots={s['slot_id']:s for s in plan['slots']};preview=set(plan['preview_slot_ids']);byid={r['slot_id']:r for r in records}
    required=[];p=OUT/'PREVIEW_REPLACEMENTS_REQUIRED.json'
    if p.exists():required=json.loads(p.read_text())['slots']
    selected=list(required);pool=[r for r in records if r['slot_id'] not in preview or r['slot_id'] in required];covered=set()
    def tokens(row):
        slot=slots[row['slot_id']];result={('group',slot['group']),('group_scene',slot['group'],row['scene_id']),('split',row['split'])}
        for key in ['small_width_band','ratio_band','gap_band','high_band']:
            if slot.get(key) is not None:result.add((slot['group'],key,slot[key]))
        for obj in row['instances']:
            T=np.array(obj['canonical8_geometry']['object_to_capture_cv_m']);v=-T[:3,:3].T@T[:3,3];axis=int(np.argmax(abs(v)))
            result.add(('view',axis,bool(v[axis]>=0)))
            if T[2,3]>=1.2:result.add(('far',1.2))
        return result
    for sid in selected:covered.update(tokens(byid[sid]))
    target=max(64,len(required))
    while len(selected)<target:
        options=[r for r in pool if r['slot_id'] not in selected]
        row=max(options,key=lambda r:100*len(tokens(r)-covered)+max(o['occlusion_inframe'] for o in r['instances']))
        selected.append(row['slot_id']);covered.update(tokens(row))
    root=OUT/'final_review_v001';root.mkdir(exist_ok=False);items=[]
    for sid in selected:
        row=byid[sid];slot=slots[sid];original=json.loads((R/row['source_candidate_record']).read_text())
        packet=root/f'INSPECTION_PACKET_{sid}.json';packet.write_text(json.dumps({'slot':slot,'record':row,'proposal':original['proposal'],'source_original_candidate':row['source_candidate_record']}))
        receipt={'stats':row['actual_phrase2_measurements'],'surface_validation':row['surface_validation']}
        items.append((slot,str(packet.relative_to(R)),receipt,str(root),'final_accepted'))
    counter=multiprocessing.Value('i',0)
    with concurrent.futures.ProcessPoolExecutor(16,initializer=pin_exporter,initargs=(counter,)) as pool:rows=list(pool.map(export_one,items,chunksize=1))
    sheets=[]
    for page in range(math.ceil(len(rows)/8)):
        canvas=Image.new('RGB',(1700,302*8),'white')
        for i,row in enumerate(rows[page*8:(page+1)*8]):
            with Image.open(R/row['row_image']) as im:canvas.paste(im,(0,i*302))
        p=root/f'FINAL_{page:02d}.jpg';canvas.save(p,quality=95);sheets.append(str(p.relative_to(R)))
    report={'status':'AWAIT_ROOT_FINAL_VISUAL_REVIEW','images':len(rows),'new_unseen_images':sum(s not in preview for s in selected),'protocol_id':audit['protocol_id'],
        'batch_identity':audit['batch_identity'],'global_identity':audit['global_identity'],'required_replaced_preview_slots':required,'sheets':sheets,'rows':rows,'scope':'New accepted images after similarity; supplements already reviewed128; no RGB rerender or model prediction.'}
    write(root/'FINAL_REVIEW_BUNDLE.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['rows','sheets']}))

if __name__=='__main__':build()
