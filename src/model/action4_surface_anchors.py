"""Provisional material seed selection from same-frame predictions only.

No GT, no automatic track confirmation, no geometry observation publication.
"""
import hashlib,json
from dataclasses import dataclass,asdict
from fractions import Fraction
from collections import Counter
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.special import expit,softmax
from data.action4_surface_snapshot import validate_snapshot
from data.action4_video_material_support import VerifiedVideoMaterialSupport
from data.action4_sam_roi_inputs import verify_image_frame,verify_native_prediction,NativeMaskIdentity,_prediction
from model.action4_material_mesh import MaterialMeshBinder
from model.action4_phase2_tracking_contracts import AnchorKey,GateScale

@dataclass(frozen=True)
class SeedPolicy:
    visible_probability:float=.7
    region_probability:float=.5
    grid_side:int=8
    max_per_cell:int=2
    max_materials:int=96
    max_geometry_trials:int=512
    spacing_short_side_fraction:float=.02
    erosion_short_side_fraction:float=.01
    def __post_init__(self):
        if any(type(getattr(self,k)) is not int for k in ('grid_side','max_per_cell','max_materials','max_geometry_trials')):raise ValueError('Integer budgets required; bool/float rejected')
        if any(isinstance(getattr(self,k),(bool,np.bool_)) for k in ('visible_probability','region_probability','spacing_short_side_fraction','erosion_short_side_fraction')):raise ValueError('Numeric policy gates cannot be bool')
        if not 0<self.visible_probability<=1 or not 0<self.region_probability<=1:raise ValueError('Explicit probability gates required')
        if (self.grid_side,self.max_per_cell)!=(8,2) or not 1<=self.max_materials<=96 or not 1<=self.max_geometry_trials<=12544:raise ValueError('Registered spatial/material budgets required')
        if not 0<self.spacing_short_side_fraction<1 or not 0<self.erosion_short_side_fraction<1:raise ValueError('Explicit normalized spacing/erosion required')


def seed_material_candidates(snapshot,frame,sam_prediction,*,expected_models,expected_sources,mesh,
        track_id,track_generation,anchor_generation,frame_id,pts,time_base,track_confirmed,
        policy=SeedPolicy(),allow_synthetic=False,frame_binding=None):
    f=verify_image_frame(frame);validate_snapshot(snapshot,frame=f,expected_models=expected_models)
    if not isinstance(mesh,MaterialMeshBinder) or not isinstance(policy,SeedPolicy):raise ValueError('Typed mesh/policy required')
    mesh.assert_sealed()
    if type(track_confirmed) is not bool or not track_confirmed:raise ValueError('Project track must be explicitly confirmed before material seeding')
    if not isinstance(track_id,str) or not track_id or any(type(v) is not int or v<0 for v in [track_generation,anchor_generation,frame_id]):raise ValueError('Explicit nonnegative track/material/frame identity required')
    if type(pts) is not int or not isinstance(time_base,str) or Fraction(time_base)<=0:raise ValueError('Integer PTS and positive rational timebase required')
    if type(allow_synthetic) is not bool:raise ValueError('Explicit engineering permission flag required')
    timing=None;video_mask=None;video_evidence=None
    if isinstance(sam_prediction,VerifiedVideoMaterialSupport):
        if allow_synthetic:raise ValueError('Real video material branch cannot use synthetic permission')
        video_mask,video_evidence=sam_prediction.consume(snapshot,f,expected_models=expected_models,
            expected_sources=expected_sources,track_id=track_id,track_generation=track_generation,
            frame_id=frame_id,pts=pts,time_base=time_base)
        timing=video_evidence['timing']
        if frame_binding is not None:
            from data.action4_video_frame_binding import validate_video_binding
            supplied=validate_video_binding(frame_binding,sam_prediction.source_frame,frame_id,pts,time_base)
            if supplied!=timing:raise ValueError('Explicit frame binding conflicts with authenticated video support')
    elif frame_binding is not None:
        from data.action4_video_frame_binding import validate_video_binding
        timing=validate_video_binding(frame_binding,f,frame_id,pts,time_base)
    elif not allow_synthetic:raise ValueError('Formal material seeds require verified source-video timing binding')
    if expected_sources['b0_model_sha256']!=expected_models['B0']['sha256']:raise ValueError('SAM and surface B0 source differ')
    m=snapshot.metadata;a=snapshot.arrays;native_id=NativeMaskIdentity(f.image_id,m['prediction_id'],f.pixel_space_id)
    base={'schema':'action4.provisional_material_seeds.v1','track_id':track_id,'track_generation':track_generation,
        'anchor_generation':anchor_generation,'image_id':f.image_id,'frame_id':frame_id,'pts':pts,'time_base':str(Fraction(time_base)),
        'surface_receipt':m['receipt'],'model_identities':expected_models,'mesh_binding_identity':mesh.binding_identity,'policy':asdict(policy),
        'material_status':'provisional_requires_multi_frame_and_appearance_validation','geometry_accepted':False,
        'query_role':'engineering_input_not_material_seed' if allow_synthetic else 'material_candidates','timing_binding':timing,'time_identity_verified':timing is not None,'appearance_stability':'not_assessed','synthetic_fixture_allowed':allow_synthetic}
    if sam_prediction is None:return dict(base,status='NO_SEEDS',reason='missing_native_SAM_support',seeds=[],counts={})
    if video_evidence is not None:
        mask=video_mask;reason='available_current_video_support'
        base['native_video_support']=video_evidence
        base['support_input_kind']='authenticated_current_video_response'
    else:
        verified=verify_native_prediction(sam_prediction,f,native_id,expected_sources,allow_synthetic=allow_synthetic)
        mask,_,reason,meta=_prediction(verified,f,native_id,expected_sources,allow_synthetic)
        base['native_SAM_cache_key']=meta['cache_key']
        base['support_input_kind']='native_still_prediction'
    base['source_SHA']=f.source_sha256;base['pixel_space_id']=f.pixel_space_id
    if mask is None:return dict(base,status='NO_SEEDS',reason=reason,seeds=[],counts={})
    ys,xs=np.nonzero(mask)
    if not len(xs):return dict(base,status='NO_SEEDS',reason='empty_native_SAM_support',seeds=[],counts={})
    short=max(1,min(int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)));radius=int(np.clip(round(policy.erosion_short_side_fraction*short),1,4))
    yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1];interior=binary_erosion(mask,structure=xx*xx+yy*yy<=radius*radius,border_value=0)
    yy,xx=np.mgrid[:112,:112];hm=np.stack([xx.ravel(),yy.ravel(),np.ones(112*112)],1);uv=(hm@np.linalg.inv(a['A112']).T)[:,:2]
    inside=(uv[:,0]>=0)&(uv[:,0]<=f.pixels.shape[1]-1)&(uv[:,1]>=0)&(uv[:,1]<=f.pixels.shape[0]-1)
    mi=np.flatnonzero(inside);native_pixel=np.rint(uv[mi]).astype(int);support=np.zeros(len(uv),bool);support[mi]=interior[native_pixel[:,1],native_pixel[:,0]]
    vis=expit(a['mask_logits'][0]).ravel();regions=softmax(a['region_logits'],axis=0).reshape(65,-1);rid=regions.argmax(0);rp=regions.max(0)
    eligible=support&(a['valid_image_support112'].ravel()>0)&(vis>=policy.visible_probability)&(rid>0)&(rp>=policy.region_probability)
    score=vis*rp;groups={}
    for k in np.flatnonzero(eligible):
        gx=min(7,int((uv[k,0]-xs.min())/max(1,xs.max()-xs.min()+1)*8));gy=min(7,int((uv[k,1]-ys.min())/max(1,ys.max()-ys.min()+1)*8))
        if gx<0 or gy<0:raise ValueError('Interior point outside registered mask bounds')
        groups.setdefault(gy*8+gx,[]).append(int(k))
    for g in groups:groups[g].sort(key=lambda k:(-float(score[k]),k))
    # Round robin gives each occupied cell a chance before revisiting it; cap
    # limits trials, never silently declares unseen candidates geometrically bad.
    ordered=[]
    for layer in range(max(map(len,groups.values()),default=0)):
        ordered.extend((g,groups[g][layer]) for g in sorted(groups) if layer<len(groups[g]))
    bbox=a['bbox_native'];diag=float(np.linalg.norm(bbox[2:]-bbox[:2]));scale_id=hashlib.sha256(json.dumps({'B0':expected_models['B0'],'image_id':f.image_id,'prediction_id':m['prediction_id'],'bbox':bbox.tolist()},sort_keys=True).encode()).hexdigest()
    scale=GateScale(diag,'full',frame_id,True,scale_id);base['gate_scale']=asdict(scale);seeds=[];counts=Counter();cell=Counter();examined=0;visited=0
    for grid,k in ordered:
        if len(seeds)>=policy.max_materials or examined>=policy.max_geometry_trials:break
        visited+=1
        if cell[grid]>=policy.max_per_cell:counts['cell_full']+=1;continue
        if any(np.linalg.norm(uv[k]-np.asarray(s['query_uv_native']))<policy.spacing_short_side_fraction*short for s in seeds):counts['spacing_rejected']+=1;continue
        examined+=1;xyz=a['low_m']+a['xyz_normalized'].reshape(3,-1)[:,k]*a['extent_m']
        binding=mesh.bind(xyz,int(rid[k]),a['rotation'],a['translation_m'],a['K_native'],uv[k],max_reprojection_px=scale.sigma)
        counts[binding['reason']]+=1
        if binding['status']!='provisional':continue
        key=AnchorKey(track_id,f"{f.image_id}:{m['prediction_id']}:surface112:{k}",anchor_generation)
        seeds.append({'key':asdict(key),'track_generation':track_generation,'query_frame_id':frame_id,'query_pts':pts,
            'query_uv_native':uv[k].tolist(),'surface_grid_index':k,'spatial_grid_id':grid,'confidence_score':float(score[k]),
            'X_cad_m':binding['X_cad_m'],'mesh_triangle_id':binding['triangle_id'],'barycentric_coordinates':binding['barycentric'],
            'XYZ_source':'A0_estimated_surface_nearest_fixed_mesh','binding_quality':binding['binding_quality'],'binding_identity':binding['binding_identity'],
            'lifecycle_state':'provisional','geometry_accepted':False})
        cell[grid]+=1
    base.update(status='PROVISIONAL_SEEDS_READY' if seeds else 'NO_SEEDS',reason='temporal_confirmation_pending' if seeds else 'no_candidate_passed_provisional_gates',seeds=seeds,
        counts=dict(counts,eligible_surface_pixels=int(eligible.sum()),geometry_trials=examined,selected=len(seeds)),
        occupied_seed_grids=len(cell),erosion_radius_native_px=radius,
        budget={'max_materials':policy.max_materials,'max_geometry_trials':policy.max_geometry_trials,'potential_candidates':len(ordered),'unvisited_candidates':len(ordered)-visited,'selection_truncated':visited<len(ordered),'budget_touched':len(seeds)>=policy.max_materials or examined>=policy.max_geometry_trials})
    return base
