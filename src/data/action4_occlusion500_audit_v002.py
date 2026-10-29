"""Final phrase2 closure and actual A0/B0 input checks; never starts training."""
from pathlib import Path
import argparse, collections, hashlib, json, os, time
import numpy as np
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R, contained_path
from data.action4_phrase2_plan import category
from data.action4_occlusion500_contract import occlusion_bin,source_eligible
from data import action4_occlusion500_queue as queue

OUT=R/'outputs/phrase2_occlusion500_v001'

def write(path, value):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)

def quantiles(values):
    return dict(zip(['min','p05','p25','p50','p75','p95','max'],map(float,np.quantile(values,[0,.05,.25,.5,.75,.95,1]))))

def audit():
    cfg=json.loads((R/'configs/data/action4_phrase2_occlusion500_v001.json').read_text())
    db=queue.connect();rows=db.execute('SELECT * FROM slots ORDER BY slot').fetchall();db.close()
    assert len(rows)==500 and all(row['status']=='accepted' for row in rows),queue.counts()
    gate=json.loads((OUT/'ROOT_PREVIEW_ACCEPTED.json').read_text())
    assert gate['decision']=='accept_preview128'
    sets={s:[] for s in cfg['split_counts']};ids=set();groups={};file_paths=set()
    position={s:collections.Counter() for s in sets};occlusion={s:{side:collections.Counter() for side in ['left','right']} for s in sets}
    scenes={s:collections.Counter() for s in sets};descriptors=[];surface_points=[];render_seconds=[];attempts=[];visible_counts=collections.Counter()
    width=[];occ=[];gap=[];height=[];leftw=[];rightw=[];roll=[];depth=[];areas=[];asset_versions=collections.Counter()
    for row in rows:
        spec=json.loads(row['spec']);r=json.loads(row['record']);split=spec['split']
        assert r['sample_id']==spec['sample_id'] and r['slot_id']==spec['slot_id'] and r['sample_id'] not in ids
        ids.add(r['sample_id']);assert r['dataset_root']=='data/phrase2' and r['split']==split
        assert r['target_count']==2 and [o['instance_id'] for o in r['instances']]==[1,2] and not r['excluded_instances']
        assert r['group_id'] not in groups or groups[r['group_id']]==split;groups[r['group_id']]=split
        assert r['status']=='training_accepted' and r['training_accepted'] and r['occlusion_mode']=='arm_geometry'
        assert r['corner_ids']==list(range(8))
        payload=json.loads(contained_path(R,r['source_candidate_record'],exists=True).read_text())
        assert payload['slot']==spec and payload['collision']['accepted'] and payload['actual_projection_error_px']<=.005
        assert len(r['surface_validation'])==2
        boxes=[];actual_outer='';side_width=[];actual_occ=[];side_roll=[];side_depth=[]
        K=np.asarray(r['camera']['K']);size=np.array([r['width'],r['height']]);cad=np.asarray(r['corners_3d_mm'])/1000
        assert tuple(size)==(3248,2464)
        paths=[r['image']]
        for obj,side,keys in zip(r['instances'],['left','right'],['AB','CD']):
            T=np.asarray(obj['canonical8_geometry']['object_to_capture_cv_m']);uv=np.asarray(obj['corners_px']);camera=cad@T[:3,:3].T+T[:3,3];h=camera@K.T
            assert uv.shape==(8,2) and np.isfinite(uv).all() and (camera[:,2]>0).all()
            assert np.max(np.abs(h[:,:2]/h[:,2:]-uv))<=.005
            assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=2e-5,rtol=0)
            assert np.isclose(np.linalg.det(T[:3,:3]),1,atol=2e-5)
            box=np.r_[uv.min(0),uv.max(0)];assert np.allclose(box,obj['bbox_full_xyxy'],atol=1e-6,rtol=0)
            assert (box[:2]>=-.5).all() and (box[2:]<=size-.5).all();boxes.append(box)
            center=(box[:2]+box[2:])/(2*size)
            assert obj['wearing_side']==side and obj['corner_ids']==list(range(8))
            assert np.allclose(obj['bbox_center_uv'],center,atol=1e-6)
            for i,key in enumerate(keys):
                klass=category(center[i],cfg['position_ranges'][key]);assert klass!='outside'
                if klass=='outer':actual_outer+=key
            av=obj['area_visible_px'];ai=obj['area_inframe_amodal_px'];af=obj['area_full_amodal_px']
            assert 0<ai<=af and av>0
            o=max(0.,1-av/ai);assert abs(o-obj['occlusion_inframe'])<1e-12
            assert abs(1-ai/af-obj['truncation'])<1e-12 and obj['truncation']<=1e-5
            b=occlusion_bin(o);assert b==spec['occlusion_bins'][side]
            source=r['occlusion_sources'][side]
            assert source_eligible(o,source['primary_source_class'])
            B=np.asarray(r['actor_pose_capture_by_side'][side]['brace_B_capture']);A=np.asarray(r['fixed_installation_by_side'][side]['A_brace_from_CAD'])
            assert np.allclose(B@A,r['actor_pose_capture_by_side'][side]['core_C_capture'],atol=1e-9,rtol=0)
            assert np.allclose(B@A,T,atol=2e-5,rtol=0)
            g=obj['canonical8_geometry'];assert g['corner_ids']==list(range(8))
            from data.action4_phrase2_visibility import canonical_visibility
            ray=obj.get('corner_ray_clear',obj['corner_visible'])
            assert np.array_equal(ray,g['ray_clear_to_virtual_point'])
            visibility=canonical_visibility(cad,T,ray,g['point_inframe'])
            if obj.get('corner_visibility_semantics')==visibility['semantics']:
                assert obj['corner_visible']==visibility['visible']
            obj.update(corner_ray_clear=visibility['ray_clear'],corner_self_visible=visibility['self_visible'],
                       corner_visible=visibility['visible'],corner_visibility_semantics=visibility['semantics'])
            g['training_corner_visible']=visibility['visible'];g['training_visibility_semantics']=visibility['semantics']
            r['corner_visibility_semantics']=visibility['semantics']
            visible_counts[sum(obj['corner_visible'])]+=1
            # Existing ray-to-virtual-box semantics is preserved, not reinterpreted
            # as a count of physically visible sharp corners on rounded CAD.
            g['training_accepted']=True;g['claim']='Same-frame fixed canonical8; host reprojection and body/mask/identity checks passed.'
            paths += [obj['mask_visible'],obj['mask_amodal_expanded'],obj['surface_supervision']['path']]
            proof=next(p for p in r['surface_validation'] if p['instance_id']==obj['instance_id'])
            assert proof['points']>0 and proof['max_plane_error_m']<=1e-6 and proof['max_reprojection_px']<=.005
            surface_points.append(proof['points'])
            w=(box[2]-box[0])/size[0];side_width.append(w);actual_occ.append(o)
            side_roll.append(r['phrase2']['by_side'][side]['forearm_roll_deg']);side_depth.append(T[2,3])
            occlusion[split][side][b]+=1
        assert actual_outer==spec['outer_coordinates'];position[split][actual_outer]+=1
        assert not (np.minimum(boxes[0][2:],boxes[1][2:])-np.maximum(boxes[0][:2],boxes[1][:2])>0).all()
        centers=np.array([(b[:2]+b[2:])/(2*size) for b in boxes]);delta=centers[1]-centers[0]
        gap.append(float(delta[0]));height.append(float(delta[1]));leftw.append(side_width[0]);rightw.append(side_width[1]);width.extend(side_width);occ.extend(actual_occ);roll.extend([(v+180)%360-180 for v in side_roll]);depth.extend(side_depth)
        descriptors.append(np.r_[centers.ravel(),side_width,actual_occ,side_roll,side_depth].round(6).tolist())
        for name in paths:
            p=contained_path(R,name,exists=True);assert p.is_relative_to(R/'data/phrase2/.staging_batches/occlusion500_v001'/split) and not p.is_symlink()
            assert name not in file_paths and p.stat().st_size>0;file_paths.add(name)
        with Image.open(R/r['image']) as image:assert image.format=='PNG' and image.size==tuple(size)
        if str(row['slot']) in gate['rgb_sha256_by_slot'] and r['rgb_sha256']!=gate['rgb_sha256_by_slot'][str(row['slot'])]:
            replacement=json.loads((OUT/'ROOT_PREVIEW_REPLACEMENT_REVIEW.json').read_text())
            assert replacement['decision']=='accept_reviewed_preview_replacements'
            assert replacement['rgb_sha256_by_slot'][str(row['slot'])]==r['rgb_sha256']
        render_seconds.append(payload['render_seconds']);attempts.append(r['attempt']);scenes[split][r['scene_id']]+=1;asset_versions[r['asset_version']]+=1;sets[split].append(r)
    for split,records in sets.items():
        assert len(records)==cfg['split_counts'][split]
        assert dict(position[split])==cfg['joint_position_quotas'][split]
        assert dict(scenes[split])==dict(zip(cfg['scene_counts'],cfg['scene_counts_by_split'][split]))
        for side in ['left','right']:assert [occlusion[split][side][i] for i in range(len(cfg['occlusion_intervals']))]==cfg['occlusion_quota_by_side_by_split'][split]
        expected={r['image'] for r in records}
        actual={str(p.relative_to(R)) for p in (R/'data/phrase2/.staging_batches/occlusion500_v001'/split/'images').iterdir()}
        assert actual==expected
    assert len({tuple(x) for x in descriptors})==500,'Duplicate geometric descriptors'
    original=json.loads((R/'outputs/phrase2_generation_v001/ORIGINAL_DATA_IDENTITY.json').read_text())
    for name,identity in original.items():
        stat=contained_path(R,name,exists=True).stat()
        assert (stat.st_size,stat.st_ino,stat.st_mtime_ns)==(identity['bytes'],identity['inode'],identity['mtime_ns']),('old data identity changed',name)
    authority=json.loads((R/json.loads((R/'configs/execution/action4_active.json').read_text())['active_ref']['path']).read_text())
    freeze=json.loads((R/authority['source_freeze']).read_text())
    asset_checks={}
    for name,a in freeze['assets'].items():
        for path,expected in [(a['metadata_path'],a['metadata_sha256']),(a['blend_ref']['path'],a['blend_ref']['sha256'])]:
            actual=hashlib.sha256(contained_path(R,path,exists=True).read_bytes()).hexdigest();assert actual==expected;asset_checks[path]=actual
    files=[contained_path(R,p,exists=True) for p in file_paths]
    storage=sum(p.stat().st_size for p in files)
    def hist2(x,y,xb,yb):return np.histogram2d(x,y,bins=[xb,yb])[0].astype(int).tolist()
    distribution={'width_fraction':quantiles(width),'width_core_fraction':float(np.mean((np.array(width)>=.08)&(np.array(width)<=.14))),
        'occlusion':quantiles(occ),'horizontal_center_delta':quantiles(gap),'vertical_center_delta':quantiles(height),'depth_m':quantiles(depth),
        'size_x_occlusion':{'width_edges':[0,.08,.11,.14,.23,1],'occlusion_edges':[.30,.39,.48,.57,.66,.75], 'counts':hist2(width,occ,[0,.08,.11,.14,.23,1],[.30,.39,.48,.57,.66,.75])},
        'gap_x_height':{'gap_edges':[-1,0,.10,.20,.34,.50,1],'height_edges':[-1,-.20,-.05,.05,.20,1],'counts':hist2(gap,height,[-1,0,.10,.20,.34,.50,1],[-1,-.20,-.05,.05,.20,1])},
        'left_x_right_width':{'edges':[0,.08,.11,.14,.23,1],'counts':hist2(leftw,rightw,[0,.08,.11,.14,.23,1],[0,.08,.11,.14,.23,1])},
        'roll_x_occlusion':{'roll_edges':[-180,-90,0,90,180],'occlusion_edges':[.30,.39,.48,.57,.66,.75],'counts':hist2(roll,occ,[-180,-90,0,90,180],[.30,.39,.48,.57,.66,.75])},
        'unique_geometric_descriptors':500,'visible_virtual_corner_counts':dict(visible_counts),'not_claimed':'Original-video measured distribution, pixel pairwise uniqueness, training accuracy, or real-video performance'}
    report={'status':'ENGINEERING_AUDIT_PASSED_FINAL_VISUAL_REVIEW_REQUIRED','images':500,'targets':1000,'corners':8000,'split_counts':{s:len(v) for s,v in sets.items()},
        'actual_joint_positions':position,'actual_occlusion_by_side':occlusion,'scenes':scenes,'asset_versions':asset_versions,'final_asset_identity':asset_checks,
        'original_manifest_identity_unchanged':True,'storage_bytes':storage,'files':len(files),'render_seconds_per_candidate':quantiles(render_seconds),'attempt_index':quantiles(attempts),
        'configuration_sha256':hashlib.sha256((R/'configs/data/action4_phrase2_occlusion500_v001.json').read_bytes()).hexdigest(),'distribution':distribution,
        'verification':'Every accepted image/mask was decoded and measured by host before acceptance. Final closure reuses bound per-frame proofs and independently checks all projections, quotas, transforms and file identities; no repeated full RGB hashing.'}
    assert int(np.sum(distribution['size_x_occlusion']['counts']))==1000
    assert int(np.sum(distribution['roll_x_occlusion']['counts']))==1000
    report['roll_histogram_semantics']='Angles wrapped modulo360 to[-180,180) for reporting only; raw GT and matrices unchanged'
    write(OUT/'FINAL_AUDIT.json',report)
    # Normalize only admission metadata; do not change image/coordinate/visibility GT.
    db=queue.connect();db.execute('BEGIN IMMEDIATE')
    for records in sets.values():
        for r in records:db.execute('UPDATE slots SET record=? WHERE slot=? AND status=\'accepted\'',(json.dumps(r),r['slot_id']))
    db.execute('COMMIT');db.close();queue.publish_manifests()
    return report

def interface_check():
    import torch
    from data.action4_occlusion500_data import surface_source,AdditionalBatchUnits
    from data.action4_stage2_data import Stage2Batches
    from data.action4_dataset import prepare_full_images
    torch.set_num_threads(1)
    provisional=OUT/'INTERFACE_GEOMETRY_ADMISSION.json'
    write(provisional,{'candidate_training_allowed':True,'scope':'Final engineering-admitted geometry, interface read validation only. No optimizer or training invoked.','audit':'outputs/phrase2_occlusion500_v001/FINAL_AUDIT.json'})
    source=surface_source();source['admission']=str(provisional.relative_to(R));results={}
    for split,n in [('train',400),('eval',50),('test',50)]:
        results[split]={}
        for role in ['A0','B0']:
            cfg={'dataset_root':'data/phrase2','role':role,'seed':2026102302,'mode':'integration','allow_unpublished_interface_validation':True,'surface_source':source,'roi_jitter':{'center_fraction':.15,'scale_range':[.65,1.35]}}
            units=AdditionalBatchUnits(cfg,split);assert len(units)==n*(2 if role=='A0' else 1)
            indices=sorted({0,len(units)//2,len(units)-1})
            shapes=[]
            for index in indices:
                item=units[(index,0,True,1)]
                if role=='A0':
                    assert item['roi'].shape==(3,448,448) and item['surface_masks'].shape==(2,112,112)
                    assert len(item['surface_sparse']['xyz'])>0 and all(torch.isfinite(v).all() for v in [item['roi'],item['surface_masks'],item['surface_sparse']['xyz']])
                    shapes.append({'index':index,'roi':list(item['roi'].shape),'sparse':len(item['surface_sparse']['xyz']),'projection_hm_px':item['metadata']['surface_projection_error_hm_px']})
                else:
                    rgb,mat=prepare_full_images([item['item']['image']],(1024,768),'cpu');assert torch.isfinite(rgb).all()
                    shapes.append({'index':index,'batch':list(rgb.shape),'targets':len(item['item']['record']['instances'])})
            sampler=Stage2Batches(len(units),32,1,0,2026102302)
            seen=[x[0] for batch in sampler for x in batch if x[2]];assert sorted(seen)==list(range(len(units)))
            results[split][role]={'units':len(units),'samples':shapes,'complete_no_drop_epoch_partition':True}
    write(OUT/'DATA_INTERFACE_VALIDATION.json',{'status':'PASS_ACTUAL_DATA_BATCHES_NO_TRAINING','results':results,'explicit_root':'data/phrase2','no_old_data_fallback':True})
    return results

if __name__=='__main__':
    from data.action4_occlusion500_similarity import accepted,require_proof
    require_proof(accepted());report=audit();interface_check()
    print(json.dumps({'audit':report['status'],'batch_interface':'PASS','formal_merge_complete':False}))
