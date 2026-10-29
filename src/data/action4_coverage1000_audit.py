"""Actual per-slot quota audit and comparable old/new geometric distributions."""
from pathlib import Path
import collections,hashlib,json,math,os,time
import numpy as np
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_coverage1000_contract import actual_accept,pair_metrics,inside,SMALL_BANDS,RATIO_BANDS,GAP_BANDS,HIGH_BANDS
from data.action4_coverage1000_similarity import base_records,new_records,fingerprint,write,OUT,STAGE
from data import action4_coverage1000_queue as queue

def distribution(records):
    widths=[];occ=[];zs=[];distances=[];ratios=[];edges=[];centers=[];dx=[];dy=[];rolls=[];joint=collections.Counter();smallhighimages=quadimages=closeimages=0
    scenes=collections.Counter();splits=collections.Counter();viewcounts=collections.Counter()
    for r in records:
        m=pair_metrics([o['bbox_full_xyxy'] for o in r['instances']],r['width']);w=m['widths'];o=[x['occlusion_inframe'] for x in r['instances']]
        near=.003<=m['d_edge']<=.03 and m['center_distance']<=.20 and not m['boxes_intersect'];smallhigh=any(.04<=a<=.08 and b>=.4 for a,b in zip(w,o))
        smaller=int(np.argmin(w));smallerhigh=.04<=w[smaller]<=.08 and o[smaller]>=.4
        closeimages+=int(near);smallhighimages+=int(smallhigh);quadimages+=int(near and smallerhigh and m['ratio']>=1.5)
        widths.extend(w);occ.extend(o);ratios.append(m['ratio']);edges.append(m['d_edge']);centers.append(m['center_distance']);dx.append(m['center_delta_width_units'][0]);dy.append(m['center_delta_width_units'][1])
        scenes[r['scene_id']]+=1;splits[r['split']]+=1
        for k,obj in enumerate(r['instances']):
            T=np.array(obj['canonical8_geometry']['object_to_capture_cv_m']);t=T[:3,3];rotation=T[:3,:3]
            zs.append(float(t[2]));distances.append(float(np.linalg.norm(t)))
            view=-rotation.T@t;view/=np.linalg.norm(view);axis=int(np.argmax(abs(view)));face=('+' if view[axis]>=0 else '-')+'XYZ'[axis];viewcounts[face]+=1
            roll=(math.degrees(math.atan2(rotation[1,0],rotation[0,0]))+180)%360-180;rolls.append(roll)
            sizebin='<4%' if w[k]<.04 else '4-5%' if w[k]<.05 else '5-6%' if w[k]<.06 else '6-7%' if w[k]<.07 else '7-8%' if w[k]<.08 else '8-14%' if w[k]<=.14 else '>14%'
            obin='<.35' if o[k]<.35 else '.35-.40' if o[k]<.4 else '.40-.50' if o[k]<.5 else '.50-.60' if o[k]<.6 else '.60-.75' if o[k]<=.75 else '>.75'
            joint[(face,sizebin,obin)]+=1
    def stats(a):
        a=np.array(a,dtype=float);return {'min':float(a.min()),'p05':float(np.quantile(a,.05)),'p50':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(a.max())}
    n=len(records);target=2*n
    hist=np.histogram2d(widths,occ,bins=([0,.04,.05,.06,.07,.08,.14,.24],[0,.35,.4,.5,.6,.75+1e-12,1.000001]))[0]
    if int(hist.sum())!=target or sum(joint.values())!=target:raise ValueError('Joint histogram omitted a target')
    return {'images':n,'targets':target,'scene_counts':dict(scenes),'split_counts':dict(splits),'width_fraction':stats(widths),'occlusion_inframe':stats(occ),
        'width_8_to_14_percent_instances':int(sum(.08<=x<=.14 for x in widths)),'small_lt_8_percent_instances':int(sum(x<.08 for x in widths)),
        'small_and_occlusion_ge_40_instances':int(sum(a<.08 and b>=.4 for a,b in zip(widths,occ))),
        'small_and_high_images':smallhighimages,'small_high_image_definition':'At least one target width in[.04,.08] inclusive and occlusion>=.4; strict<.08 instance count separately retained for baseline comparison.',
        'ratio_ge_1p5_images':int(sum(x>=1.5 for x in ratios)),'near_pair_images':closeimages,'four_conditions_images':quadimages,
        'four_conditions_definition':'The smaller camera itself has width[.04,.08] and high occlusion; R>=1.5 and both2D near-pair conditions. A high-occluded larger camera cannot substitute.',
        'width_by_wearing_side':{s:stats(np.array(widths).reshape(-1,2)[:,i]) for i,s in enumerate(['left','right'])},
        'left_right_width_correlation':float(np.corrcoef(np.array(widths).reshape(-1,2).T)[0,1]),
        'larger_wearing_side_counts':dict(collections.Counter('left' if a>b else 'right' if b>a else 'tie' for a,b in np.array(widths).reshape(-1,2))),
        'size_ratio':stats(ratios),'edge_gap_W':stats(edges),'center_distance_W':stats(centers),'horizontal_delta_W':stats(dx),'vertical_delta_W':stats(dy),
        'absolute_horizontal_center_gap_W':stats(np.abs(dx)),'absolute_vertical_center_gap_W':stats(np.abs(dy)),
        'optical_Z_m':stats(zs),'euclidean_CAD_origin_distance_m':stats(distances),'far_target_counts':{'Z_ge_080':sum(z>=.8 for z in zs),'Z_ge_100':sum(z>=1 for z in zs),'Z_120_to_140':sum(1.2<=z<=1.4 for z in zs)},
        'canonical_view_dominant_axis':dict(viewcounts),'camera_ZYX_z_angle_deg_wrapped':stats(rolls),
        'observation_direction_x_size_x_occlusion':[{'view':a,'size':b,'occlusion':c,'targets':v} for (a,b,c),v in sorted(joint.items())],
        'size_x_occlusion_counts':hist.astype(int).tolist(),'geometry_measurement':'Actual full-body bbox / W; actual visible/amodal area; Z and norm of fixed CAD-origin translation; canonical view=-R.T@t.',
        'roll_semantics':'Wrapped CV ZYX z-angle of body R, not interchangeable with forearm-axis roll; both per-frame actual transforms and forearm controls retained.'}

def audit():
    if queue.counts()!={'accepted':1000}:raise ValueError('The new batch is not complete')
    cfg=json.loads((R/'configs/data/action4_coverage1000_v001.json').read_text());plan=json.loads((OUT/'SLOTS.json').read_text());spec={s['slot_id']:s for s in plan['slots']}
    records=new_records();old,commit=base_records()
    if len(records)!=1000 or {r['slot_id'] for r in records}!=set(spec):raise ValueError('Final accepted slot set differs')
    actual=collections.Counter();expected=collections.Counter();paths=set();bytes_total=0;max_surface=max_projection=0.;point_counts=[]
    for slot in spec.values():expected[(slot['group'],slot['split'],slot['scene_id'],slot.get('small_width_band'),slot.get('ratio_band'),slot.get('gap_band'),slot.get('high_band'),slot['designated_side'])]+=1
    for row in records:
        slot=spec[row['slot_id']];ok,why,metrics=actual_accept(slot,row)
        if not ok:raise ValueError(('Actual quota failed',row['sample_id'],why,metrics))
        if slot['group'] in 'BE' and slot['high_required'] and slot['high_side']!=slot['designated_side']:raise ValueError('High occlusion must belong to the smaller camera')
        if row['coverage_protocol_id']!=cfg['protocol_id'] or row['corner_ids']!=list(range(8)):raise ValueError('Wrong protocol or IDs')
        DataAdapter(R).validate_sample(row,check_image=False)
        if row['pending_host_png_validation'] or not row['training_accepted'] or len(row['surface_validation'])!=2:raise ValueError('Missing completed per-frame host validation')
        for surface,obj in zip(row['surface_validation'],row['instances']):
            if surface['instance_id']!=obj['instance_id'] or surface['points']<=0 or surface['max_reprojection_px']>.005 or surface['max_plane_error_m']>1e-6:raise ValueError('Invalid sparse geometry receipt')
            max_surface=max(max_surface,surface['max_plane_error_m']);max_projection=max(max_projection,surface['max_reprojection_px']);point_counts.append(surface['points'])
            if obj['wearing_side']!=['left','right'][obj['instance_id']-1] or obj['corner_ids']!=list(range(8)):raise ValueError('Instance identity re-ranked by image coordinates')
            if obj['occlusion_inframe']>.35 and row['occlusion_sources'][obj['wearing_side']]['primary_occluder_object']==obj['wearing_side']+'_forearm_continuation':raise ValueError('Upper-arm occlusion substituted for requested wrist/forearm')
        names=[row['image']]+[name for obj in row['instances'] for name in [obj['mask_visible'],obj['mask_amodal_expanded'],obj['surface_supervision']['path']]]
        if len(set(names))!=7:raise ValueError('Wrong artifact count')
        for name in names:
            p=contained_path(R,name,exists=True)
            if not p.is_relative_to(STAGE/row['split']) or p.is_symlink() or name in paths:raise ValueError('Unowned or duplicated artifact')
            paths.add(name);bytes_total+=p.stat().st_size
        actual[(slot['group'],slot['split'],slot['scene_id'],slot.get('small_width_band'),slot.get('ratio_band'),slot.get('gap_band'),slot.get('high_band'),slot['designated_side'])]+=1
    if actual!=expected:raise ValueError('Final joint slot quota count differs')
    summaries={'old5500':distribution(old),'new1000':distribution(records),'merged6500':distribution(old+records)}
    new=summaries['new1000']
    if new['ratio_ge_1p5_images']<600 or new['near_pair_images']<600 or new['small_and_high_images']<400 or new['four_conditions_images']<200:raise ValueError('Actual joint coverage minima not met')
    far=new['far_target_counts']
    if far['Z_ge_080']<100 or far['Z_ge_100']<30 or far['Z_120_to_140']<10:raise ValueError('Actual far coverage not met')
    group_reports={}
    for g in 'ABCDE':
        selected=[r for r in records if spec[r['slot_id']]['group']==g];slots=[spec[r['slot_id']] for r in selected]
        if len(selected)!=200:raise ValueError('Group count differs')
        group_reports[g]={'images':200,'split':dict(collections.Counter(s['split'] for s in slots)),'scene':dict(collections.Counter(s['scene_id'] for s in slots)),
            'designated_side':dict(collections.Counter(s['designated_side'] for s in slots)),
            'small_width_bands':{str(k):sum(s.get('small_width_band')==k for s in slots) for k in range(4)} if g in 'ABCE' else None,
            'ratio_bands':{str(k):sum(s.get('ratio_band')==k for s in slots) for k in range(3)} if g in 'BDE' else None,
            'gap_bands':{str(k):sum(s.get('gap_band')==k for s in slots) for k in range(3)} if g in 'CDE' else None,
            'high_occlusion_bands':{str(k):sum(s.get('high_band')==k for s in slots) for k in range(3)},'actual':distribution(selected)}
        physical=[pair_metrics([o['bbox_full_xyxy'] for o in row['instances']],row['width']) for row in selected]
        measured={}
        for key,bands,values in [
            ('small_width',SMALL_BANDS,[m['widths'][['left','right'].index(s['designated_side'])] for s,m in zip(slots,physical)] if g in 'ABCE' else []),
            ('ratio',RATIO_BANDS,[m['ratio'] for m in physical] if g in 'BDE' else []),
            ('gap',GAP_BANDS,[m['d_edge'] for m in physical] if g in 'CDE' else []),
            ('designated_high_occlusion',HIGH_BANDS,[row['instances'][['left','right'].index(s['high_side'])]['occlusion_inframe'] for row,s in zip(selected,slots) if s['high_required']])]:
            measured[key]={str(k):sum(inside(v,band,k==len(bands)-1) for v in values) for k,band in enumerate(bands)}
            if sum(measured[key].values())!=len(values):raise ValueError('Actual measured strata omitted a value')
        group_reports[g]['actual_measured_strata']=measured
    report={'status':'PASS_ACTUAL1000_QUOTA_FILE_AND_HOST_GEOMETRY_CLOSURE','images':1000,'targets':2000,'fixed_id_corners':16000,
        'protocol_id':cfg['protocol_id'],'batch_identity':fingerprint(records),'global_identity':fingerprint(old+records),
        'configuration_sha256':hashlib.sha256((R/'configs/data/action4_coverage1000_v001.json').read_bytes()).hexdigest(),'groups':group_reports,
        'distributions':summaries,'artifacts':len(paths),'bytes':bytes_total,'max_surface_plane_error_m':max_surface,'max_surface_reprojection_px':max_projection,
        'sparse_surface_points_min':min(point_counts),'sparse_surface_points_max':max(point_counts),'old5500_manifest_identity':commit['manifest_sha256'],
        'per_frame_full_pixel_hash_and_mask_validation':'Reused immutable accepted host receipts; no redundant full-library hashing.',
        'remaining':'Global similarity, final visual review, actual integration inputs, consumer conflict check and atomic publication.',
        'training_started':False,'time_unix':time.time()}
    write(OUT/'FINAL_AUDIT.json',report);write(OUT/'DISTRIBUTION_REPORT.json',summaries)
    write(OUT/'INTERFACE_GEOMETRY_ADMISSION.json',{'candidate_training_allowed':True,'formal_training_allowed':False,'purpose':'Integration-only A0 mask/XYZ batch validation, not training authorization',
        'protocol_id':cfg['protocol_id'],'batch_identity':report['batch_identity'],'audit':'outputs/coverage1000_v001/FINAL_AUDIT.json'})
    queue.publish_manifests();return report

if __name__=='__main__':
    result=audit();print(json.dumps({k:v for k,v in result.items() if k not in ['groups','distributions']},indent=2))
