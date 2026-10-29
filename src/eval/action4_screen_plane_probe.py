"""Two clear-frame screen-region geometry hypotheses; no model/data/default mutation."""
import json,hashlib
from pathlib import Path
import numpy as np
from data.action4_paths import PROJECT_ROOT as ROOT,contained_path
from model.action4_planar_landmark_pose import planar_pose_candidates


def run(output):
    import cv2
    cv2.setNumThreads(1)
    out=contained_path(ROOT,output,exists=False);out.mkdir(exist_ok=False)
    seed_ref=contained_path(ROOT,'outputs/action4_sealed7300_ray_seeds_v003/RESULTS.json',exists=True)
    seeds=json.loads(seed_ref.read_text())
    if seeds.get('schema')!='action4.frame4_provisional_ray_seeds.v1' or len(seeds.get('rows',[]))!=2 or {x['track_id'] for x in seeds['rows']}!={'track_00','track_01'} or any(x['pose']['frame_id']!=4 for x in seeds['rows']):raise ValueError('Exact dual frame4 seed source required')
    policy=json.loads((ROOT/'configs/model/action4_material_mesh_policy_v001.json').read_text())
    mesh_path=contained_path(ROOT,policy['mesh'],exists=True)
    if hashlib.sha256(mesh_path.read_bytes()).hexdigest()!=policy['mesh_sha256']:raise ValueError('Frozen mesh changed')
    mesh=np.load(mesh_path,allow_pickle=False);vertices=mesh['vertices_mm']/1000.;faces=mesh['faces'];parts=mesh['face_components']
    screen=vertices[np.unique(faces[parts==6])];low=screen.min(0);high=screen.max(0);center=(low+high)/2;center[1]=float(np.median(screen[:,1]))
    canonical_path=contained_path(ROOT,policy['canonical_manifest'],exists=True)
    if hashlib.sha256(canonical_path.read_bytes()).hexdigest()!=policy['canonical_manifest_sha256']:raise ValueError('Frozen canonical changed')
    canonical=np.asarray(json.loads(canonical_path.read_text())['corners_3d_mm'])/1000.
    diagonal=2.25*25.4;wh=np.array([640.,360.]);nominal=diagonal*wh/np.linalg.norm(wh)/1000.
    definitions=[('mesh_component6_envelope_approx_plane',(high-low)[[0,2]]),('nominal_active_display_center_assumed',nominal)]
    geometry=dict(component6_extent_mm=((high-low)*1000).tolist(),component6_planar_deviation_mm=float(np.ptp(screen[:,1])*1000),
       nominal_active_mm=(nominal*1000).tolist(),official_source='https://www.dji.com/osmo-action-4/specs',
       source_values=dict(diagonal_inches=2.25,resolution_pixels=[640,360],ppi=326),
       caveat='Nominal active size derived from diagonal/aspect. Plane location/center and region semantics are not physically measured. Neither definition is GT.')
    results=[]
    for row in seeds['rows']:
        tid=row['track_id'];ref=row['support'];path=contained_path(ROOT,ref['path'],exists=True)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('Mask support changed')
        arrays=np.load(path,allow_pickle=False);dynamic=arrays['dynamic'];body=arrays['body']
        n,labels,stats,_=cv2.connectedComponentsWithStats(dynamic.astype(np.uint8),8)
        if n<2:results.append(dict(track_id=tid,status='no_region'));continue
        largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));dominant=labels==largest
        contour=max(cv2.findContours(dominant.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea)
        quads=[];seen=set();arc=cv2.arcLength(contour,True)
        for eps in [.005,.01,.015,.02,.03,.04]:
            q=cv2.approxPolyDP(contour,eps*arc,True)
            if len(q)!=4 or not cv2.isContourConvex(q):continue
            q=q.reshape(4,2);key=tuple(sorted(map(tuple,q.tolist())))
            if key in seen:continue
            seen.add(key);qmask=np.zeros(dynamic.shape,np.uint8);cv2.fillConvexPoly(qmask,q,1);qm=qmask>0
            quads.append(dict(quad_id=len(quads),epsilon_fraction=eps,uv_native=q.astype(float).tolist(),
                 raw_region_IoU=float((qm&dynamic).sum()/(qm|dynamic).sum()),
                 raw_region_coverage=float((qm&dynamic).sum()/dynamic.sum())))
        candidates=[];K=np.asarray(row['pose']['K'])
        for quad in quads:
            pixels=np.asarray(quad['uv_native'])
            for definition,size in definitions:
                w,h=size;cx,cy,cz=center
                xyz=np.array([[cx-w/2,cy,cz+h/2],[cx+w/2,cy,cz+h/2],[cx+w/2,cy,cz-h/2],[cx-w/2,cy,cz-h/2]])
                for reverse in [False,True]:
                    p=pixels[::-1] if reverse else pixels
                    for shift in range(4):
                        image=np.roll(p,shift,axis=0)
                        answer=planar_pose_candidates(xyz,image,K,canonical)
                        for candidate in answer['candidates']:
                            rec=dict(candidate,quad_id=quad['quad_id'],definition=definition,reverse=reverse,cyclic_shift=shift,
                                geometry_accepted=False,physical_corner_order_verified=False)
                            if candidate['valid']:
                                R=np.array(candidate['R']);t=np.array(candidate['t_m']);camera=-R.T@t
                                facing=bool(np.dot(np.array([0,1,0]),camera-center)>0);uv=np.array(candidate['corners_native'])
                                hull=cv2.convexHull(uv.astype(np.float32)).reshape(-1,2);polygon=np.zeros(body.shape,np.uint8)
                                cv2.fillConvexPoly(polygon,np.rint(hull).astype(np.int32),1);pm=polygon>0
                                rec.update(front_facing=facing,body_envelope_IoU=float((pm&body).sum()/(pm|body).sum()),body_mask_coverage=float((pm&body).sum()/body.sum()))
                            candidates.append(rec)
        ranked=sorted([x for x in candidates if x['valid'] and x['front_facing']],key=lambda x:(-x['body_envelope_IoU'],x['correspondence_rms_native_px']))
        results.append(dict(track_id=tid,frame_id=4,mask_source=ref,raw_mask_pixels=int(dynamic.sum()),dominant_component_pixels=int(dominant.sum()),
            unassigned_island_pixels=int(dynamic.sum()-dominant.sum()),raw_pixels_preserved=True,quads=quads,candidates=candidates,
            ranking_purpose='current body support diagnostic only, not GT selection',ranked_candidate_indices=[candidates.index(x) for x in ranked],
            best_by_definition={name:next((x for x in ranked if x['definition']==name),None) for name,_ in definitions},
            status='HYPOTHESES_ONLY_PHYSICAL_REGION_AND_ORDER_UNVERIFIED'))
    report=dict(status='CLEAR_FRAME_SCREEN_GEOMETRY_DIAGNOSTIC_NOT_ACCEPTANCE',geometry=geometry,results=results,
        input_seed_results=dict(path=str(seed_ref.relative_to(ROOT)),sha256=hashlib.sha256(seed_ref.read_bytes()).hexdigest()),
        CAD_changed=False,new_model_calls=0,GT_used=False,default_changed=False,camera_assumption='project estimate pinhole, not verified calibration')
    (out/'RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False));return report
