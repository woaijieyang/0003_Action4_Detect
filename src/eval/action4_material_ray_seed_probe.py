"""Frame4-only RGB corners with estimated SAM80 pose first-mesh-hit XYZ.

No model inference, PnP, tracking, learning, trusted pose or rigid acceptance.
"""
import hashlib,json,os,subprocess,difflib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path('/data/zy/vln/a-test');OUT=ROOT/'outputs/action4_material_ray_seed_probe_v001'
def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def select_grid(points,short):
    groups=defaultdict(list)
    for p in points:
        if p['status']=='provisional_fixed_binding':groups[p['spatial_grid_id']].append(p)
    for group in groups.values():group.sort(key=lambda p:(-p['response'],p['feature_id']))
    ordered=[groups[g][i] for i in range(max(map(len,groups.values()),default=0)) for g in sorted(groups) if i<len(groups[g])]
    selected=[];cells=Counter();counts=Counter()
    for p in ordered:
        reason='max96' if len(selected)>=96 else 'grid_capacity2' if cells[p['spatial_grid_id']]>=2 else 'spacing_0p02_short' if any(np.linalg.norm(np.asarray(p['native_uv'])-q['native_uv'])<.02*short for q in selected) else 'selected_provisional'
        p['selection_reason']=reason;counts[reason]+=1
        if reason=='selected_provisional':selected.append(p);cells[p['spatial_grid_id']]+=1
    return selected,dict(cells),dict(counts)

def run(output=None,dynamic_hull_mode="global",input_lock_ref=None,surface_index_ref=None,feature_quality=.01):
    if not np.isfinite(feature_quality) or not 0 < feature_quality <= 1:raise ValueError("Finite feature proposal quality in (0,1] required")
    import cv2
    from PIL import Image,ImageDraw,ImageFont
    from scipy.ndimage import binary_erosion
    from data.action4_paths import contained_path
    from data.action4_surface_snapshot import load_snapshot
    from data.action4_sam_roi_inputs import array_digest
    from eval.action4_roi_sweep import read_binding,binding
    from eval.action4_reanchor_roi_provider import ReanchorROIProvider
    from eval.action4_sam_full_review import ReceiptVerifier
    from model.action4_face_motion_evidence import FaceMotionConfig,border_support
    from model.action4_single_frame_pose import POLICY
    from model.action4_surface_anchors import SeedPolicy
    OUT=contained_path(ROOT,output or 'outputs/action4_material_ray_seed_probe_v001',exists=False)
    if set(os.sched_getaffinity(0))!={58,59} or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU58-59/GPU-hidden required')
    cv2.setNumThreads(1);sp=SeedPolicy();OUT.mkdir(exist_ok=True)
    lockref=input_lock_ref or binding('outputs/action4_sam80_fixed_video_v001/INPUT_LOCK.json');lock=read_binding(lockref);savedref=surface_index_ref or binding('outputs/action4_sam80_saved_face_inputs_v001/COMPLETE.json');saved=read_binding(savedref);refs=[r for r in saved['results'] if r['frame_id']==4]
    if len(refs)!=2 or {r['track_id'] for r in refs}!={'track_00','track_01'}:raise ValueError('Frame4 dual source required')
    provider=ReanchorROIProvider(**lock['video']['track_replay'],register_project_bodies=True);bodies,receipt=provider.current_native_masks(4);frame=read_binding(lock['video']['frames_manifest'])['frames'][4];image=contained_path(ROOT,frame['image'],exists=True)
    if binding(image)['sha256']!=frame['image_sha256'] or receipt['RGB_sha256']!=frame['image_sha256'] or receipt['pts']!=frame['pts']:raise ValueError('Frame identity changed')
    with Image.open(image) as im:rgb=np.asarray(im.convert('RGB'))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);responses=cv2.cornerMinEigenVal(gray,blockSize=7);meshpolicy=read_binding(binding('configs/model/action4_material_mesh_policy_v001.json'));verifier=ReceiptVerifier();rows=[];expected={k:{a:v[a] for a in ('path','sha256')} for k,v in saved['models'].items()}
    for ref in refs:
        track=ref['track_id'];surface_path=contained_path(ROOT,ref['surface']['path'],exists=True)
        if binding(surface_path)!=ref['surface']:raise ValueError('Surface changed')
        surface=load_snapshot(surface_path,expected_models=expected);m=surface.metadata;a=surface.arrays;v=m['video_context'];p=v['candidate'];reg=v['source_ledger']['project_body_registrations'][track]
        if v['frame_id']!=4 or p['track_id']!=track or p['pts']!=frame['pts'] or m['source_sha256']!=frame['image_sha256'] or m['sam_input']['current_native_receipt']!=receipt:raise ValueError('SAM80 pose source differs')
        body=bodies[track];other=bodies['track_01' if track=='track_00' else 'track_00'];native=ROOT/'outputs/action4_face_native_seed_pilot_v003'/track/'run/face_03'
        face=read_binding(binding(native/'FACE_SEED.json'));decision=read_binding(binding(native/'SEED_DECISION.json'));snapref=binding(native/'SNAPSHOT.json');snap=read_binding(snapref)['snapshot'];request=read_binding(binding(native/'PROMPT_REQUEST.json'));complete=read_binding(binding(native.parent/'FACE_PILOT_COMPLETE.json'));entry=next(x for x in complete['results'] if x['face_id']==3)
        if entry['snapshot_sha256']!=snapref['sha256'] or entry['seed_decision']!=decision:raise ValueError('Native transaction changed')
        if (face['global_frame_id']!=4 or face['pts']!=frame['pts'] or face['local_index']!=0 or face['rgb_sha256']!=frame['image_sha256'] or face['time_base']!=receipt['time_base'] or face['pixel_space_id']!=m['pixel_space_id'] or not np.array_equal(np.asarray(face['K_native']).reshape(3,3),a['K_native']) or face['body_track_id']!=reg['scoped_body_id'] or face['body_generation']!=reg['project_generation']):raise ValueError('Display region frame/PTS/owner differs')
        if decision['face_key']!=[face['body_track_id'],face['body_generation'],3,face['face_generation']] or decision['status']!='predicted_face_support_only':raise ValueError('Region owner decision differs')
        first=next(x for x in snap['frames'] if x['global_frame_id']==4)
        if first['pts']!=frame['pts'] or first['local_index']!=0 or first['status']!='target_present' or snap['session_id']!=request['session_id'] or snap['obj_id']!=decision['native_obj_id'] or snap['region_identity']!=face['region_identity']:raise ValueError('Region session/time differs')
        raw=verifier.raw(read_binding(decision['raw_receipt_ref']),native);index=decision['selected_raw_index'];dynamic=np.asarray(raw['outputs']['out_binary_masks'])[index]
        if raw['frame_index']!=0 or int(np.asarray(raw['outputs']['out_obj_ids'])[index])!=decision['native_obj_id'] or dynamic.dtype!=np.bool_ or dynamic.shape!=body.shape or not dynamic.any():raise ValueError('Actual native display response invalid')
        if complete['face_pilot_inputs']['body_mask_ref']['sha256']!=ref['body_mask']['sha256']:raise ValueError('Old/current body identity differs')
        _,excluded=border_support(body,dynamic,FaceMotionConfig(dynamic_hull_mode=dynamic_hull_mode));y,x=np.nonzero(body);short=int(max(1,min(x.max()-x.min()+1,y.max()-y.min()+1)));radius=int(np.clip(round(sp.erosion_short_side_fraction*short),1,4));yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1];interior=binary_erosion(body,structure=xx*xx+yy*yy<=radius*radius,border_value=0);allowed=interior&~other&~excluded
        camera=a['canonical_m']@a['rotation'].T+a['translation_m'];h=camera@a['K_native'].T;uv=h[:,:2]/h[:,2:];box=np.r_[uv.min(0),uv.max(0)];coverage=float(((x>=box[0])&(x<=box[2])&(y>=box[1])&(y<=box[3])).mean());proper=bool(np.allclose(a['rotation'].T@a['rotation'],np.eye(3),atol=1e-4,rtol=0) and abs(np.linalg.det(a['rotation'])-1)<=1e-4);compatible=proper and bool((camera[:,2]>1e-4).all()) and coverage>=POLICY['coarse_support_coverage_min']
        detected=cv2.goodFeaturesToTrack(gray,maxCorners=0,qualityLevel=feature_quality,minDistance=1,mask=body.astype(np.uint8)*255,blockSize=7);features=[];counts=Counter()
        for fid,point in enumerate([] if detected is None else detected[:,0].tolist()):
            ix,iy=np.rint(point).astype(int);reason='body_erosion_boundary' if not interior[iy,ix] else 'other_body_exclusion' if other[iy,ix] else 'LCD_convex_hull_plus3px' if excluded[iy,ix] else 'initial_pose_mask_incompatible' if not compatible else 'eligible_for_ray';counts[reason]+=1;gx=min(7,int((point[0]-x.min())/(x.max()-x.min()+1)*8));gy=min(7,int((point[1]-y.min())/(y.max()-y.min()+1)*8))
            features.append({'feature_id':fid,'native_uv':point,'response':float(responses[iy,ix]),'spatial_grid_id':int(gy*8+gx),'image_gate_reason':reason,'status':'pending_ray' if reason=='eligible_for_ray' else 'image_rejected'})
        support=OUT/(track+'_SUPPORT.npz');np.savez_compressed(support,body=body,other=other,eroded=interior,dynamic=dynamic,LCD_exclusion=excluded,allowed=allowed)
        pose={'surface':ref['surface'],'surface_receipt':m['receipt'],'models':m['models'],'R':a['rotation'].tolist(),'t_m':a['translation_m'].tolist(),'K':a['K_native'].tolist(),'bbox_for_a0':p['bbox_for_a0'],'roi_decision_id':p['roi_decision_id'],'frame_id':4,'pts':frame['pts'],'RGB_sha256':frame['image_sha256']};pose['identity']=identity(pose)
        rows.append({'track_id':track,**{k:pose[k] for k in ('R','t_m','K')},'native_uv':[p['native_uv'] for p in features if p['status']=='pending_ray'],'ray_feature_ids':[p['feature_id'] for p in features if p['status']=='pending_ray'],'features':features,'pose':pose,'registration':reg,'short_side_px':short,'erosion_radius_px':radius,'raw_feature_count':len(features),'image_gate_counts':dict(counts),'support':binding(support),'compatibility':{'coverage':coverage,'threshold':POLICY['coarse_support_coverage_min'],'positive_depth':bool((camera[:,2]>1e-4).all()),'proper_R':proper,'passed':compatible,'trusted_pose':False},'support_pixel_counts':{'body':int(body.sum()),'eroded':int(interior.sum()),'after_other':int((interior&~other).sum()),'after_LCD':int(allowed.sum())},'native_display_region':{'seed':binding(native/'FACE_SEED.json'),'decision':binding(native/'SEED_DECISION.json'),'raw_receipt':decision['raw_receipt_ref'],'snapshot':snapref,'mask_sha256':array_digest(dynamic),'old_A0_sha256':face['a0_model_sha256'],'use':'same_RGB_exclusion_only_not_CAD_face_or_pose'}})
    spec={'mesh':{'path':meshpolicy['mesh'],'sha256':meshpolicy['mesh_sha256']},'mesh_policy':binding('configs/model/action4_material_mesh_policy_v001.json'),'excluded_components':meshpolicy['excluded_components'],'rows':rows,'frame':frame,'input_lock':lockref,'surface_index':savedref,'body_receipt':receipt,'feature_policy':{'detector':'goodFeaturesToTrack','maxCorners':0,'qualityLevel':feature_quality,'minDistance':1,'blockSize':7,'raw_scope':'current_native_body','LCD_exclusion':'existing border_support hull plus3px','dynamic_hull_mode':dynamic_hull_mode,'grid_side':8,'max_per_cell':2,'max_materials':96,'spacing_short_side_fraction':.02},'geometry_accepted':False,'initial_pose_trusted':False,'GT_used':False}
    (OUT/'RAY_INPUT.json').write_text(json.dumps(spec,indent=2,allow_nan=False))
    oldpath=ROOT/'outputs/action4_face_texture_probe_v001/ray_bind.py';original=oldpath.read_text();old='outputs/action4_face_texture_probe_v001';new=str(OUT.relative_to(ROOT))
    if original.count(old)!=1:raise ValueError('Unexpected old ray entry path contract')
    raycode=original.replace(old,new).replace('sys.path.insert(0,str(ROOT/"src"))','sys.path.insert(0,'+repr(str(Path(__file__).resolve().parents[1]))+')');raypath=OUT/'ray_bind_frozen.py';raypath.write_text(raycode);raypath.chmod(0o444);(OUT/'RAY_COPY.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),raycode.splitlines(True),fromfile=str(oldpath),tofile=str(raypath))))
    binary=ROOT/'runtime/blender-3.6.23-linux-x64/blender'
    if not binary.resolve().is_relative_to(ROOT):raise ValueError('Blender outside project')
    runtime=OUT/'runtime';runtime.mkdir(exist_ok=True);env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TMPDIR=str(runtime),XDG_CACHE_HOME=str(runtime),BLENDER_USER_CONFIG=str(runtime),BLENDER_USER_SCRIPTS=str(runtime),BLENDER_USER_DATAFILES=str(runtime))
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    (OUT/'EXECUTION.json').write_text(json.dumps({'main_pid':os.getpid(),'original_ray_source':binding(oldpath),'ray_copy':binding(raypath),'intersection_source':binding(Path(__file__).resolve().parents[1]/'data/action4_surface_intersection.py'),'blender_path':str(binary.relative_to(ROOT)),'CPU_affinity':sorted(os.sched_getaffinity(0)),'GPU':False,'new_model_calls':0,'PnP_started':False,'CoTracker_started':False},indent=2))
    with (OUT/'BLENDER.log').open('x') as log:result=subprocess.run([str(binary),'-b','-t','1','--factory-startup','--python',str(raypath)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Existing Blender ray entry failed; log retained')
    rayresult=json.loads((OUT/'FIXED_RAY_BINDINGS.json').read_text());summary=[];tiles=[];font=ImageFont.load_default(size=17)
    for row,rayrow in zip(rows,rayresult['rows']):
        if row['track_id']!=rayrow['seed']['track_id'] or len(rayrow['points'])!=len(row['ray_feature_ids']):raise ValueError('Ray feature domain changed')
        counts=Counter()
        for fid,hit in zip(row['ray_feature_ids'],rayrow['points']):
            p=row['features'][fid];p.update({k:v for k,v in hit.items() if k not in ('keypoint_id','native_uv')});reason=spec['excluded_components'].get(str(p.get('component_id')))
            if reason:p.update(status='rejected',reason='excluded_component:'+reason)
            p.update(source_pose_identity=row['pose']['identity'],geometry_accepted=False,material_stability='unknown',reflection_strap_mount_stability='unknown')
            if 'triangle_id' in p:p['binding_identity']=identity({'mesh':spec['mesh'],'pose':p['source_pose_identity'],'UV':p['native_uv'],'triangle':p['triangle_id'],'barycentric':p['barycentric'],'XYZ':p['X_cad_m']})
            counts[p['reason']]+=1
        selected,cells,selection=select_grid(row['features'],row['short_side_px']);row.update(ray_gate_counts=dict(counts),selection_counts=selection,selected_feature_ids=[p['feature_id'] for p in selected],selected_count=len(selected),occupied_grids=cells,occupied_grid_count=len(cells),at_least12_and4grids=len(selected)>=12 and len(cells)>=4)
        with np.load(ROOT/row['support']['path']) as z:body=z['body'];allowed=z['allowed'];excluded=z['LCD_exclusion']
        y,x=np.nonzero(body);canvas=rgb.copy();canvas[excluded]=(canvas[excluded]*.6+np.array([255,0,0])*.4).astype(np.uint8);im=Image.fromarray(canvas);d=ImageDraw.Draw(im)
        for p in selected:
            u,v=p['native_uv'];ix,iy=np.rint([u,v]).astype(int);assert allowed[iy,ix] and str(p['component_id']) not in spec['excluded_components'];d.ellipse([u-4,v-4,u+4,v+4],fill='lime');d.text((u+5,v-8),str(p['feature_id']),font=font,fill='yellow')
        im=im.crop((max(0,x.min()-25),max(0,y.min()-25),min(rgb.shape[1],x.max()+26),min(rgb.shape[0],y.max()+26)));im.thumbnail((900,650));tile=Image.new('RGB',(900,700),'black');tile.paste(im,(0,45));d=ImageDraw.Draw(tile);d.text((8,8),f"{row['track_id']} provisional {len(selected)} / grids {len(cells)} | red=LCD exclusion",font=font,fill='white');tiles.append(tile)
        summary.append({k:row[k] for k in ('track_id','raw_feature_count','image_gate_counts','ray_gate_counts','selection_counts','selected_count','occupied_grid_count','occupied_grids','at_least12_and4grids','compatibility')})
    final={'schema':'action4.frame4_provisional_ray_seeds.v1','rows':rows,'input':binding(OUT/'RAY_INPUT.json'),'ray_source':binding(OUT/'FIXED_RAY_BINDINGS.json'),'rebinding_allowed':False,'geometry_accepted':False,'initial_pose_trusted':False,'GT_used':False,'PnP_started':False,'CoTracker_started':False,'new_model_calls':0,'reflection_strap_mount_stability':'unknown'};(OUT/'RESULTS.json').write_text(json.dumps(final,indent=2,allow_nan=False))
    contact=Image.new('RGB',(1800,700));contact.paste(tiles[0],(0,0));contact.paste(tiles[1],(900,0));contact.save(OUT/'FRAME4_PROVISIONAL_RAY_SEEDS.jpg',quality=94)
    (OUT/'COMPLETE.json').write_text(json.dumps({'status':'FRAME4_DUAL_PROVISIONAL_RAY_SEEDS_COMPLETE_NOT_RIGID_ACCEPTANCE','summary':summary,'results':binding(OUT/'RESULTS.json'),'visual':binding(OUT/'FRAME4_PROVISIONAL_RAY_SEEDS.jpg'),'frame_count':1,'instances':2,'new_model_calls':0,'PnP_started':False,'CoTracker_started':False,'geometry_accepted':False,'surface_route_unchanged':True,'caveat':'XYZ inherits SAM80 A0 pose bias. Old native LCD region is exclusion only, not CAD-face truth. Reflection/strap/mount stability unknown; multiframe confirmation not run.'},indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':run()
