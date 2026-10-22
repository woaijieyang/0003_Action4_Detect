"""Finite image-layout-conditioned translation/depth proposals; never3D GT or scale.

Frozen same-side source rotations are enumerated, not optimized. Only capture
translation changes. The consumer has an explicit separate mode for these
larger moves; old4mm/8degree sampling limits remain intact.
"""
from pathlib import Path
from copy import deepcopy
import hashlib,json,math,io,re
from functools import lru_cache
import numpy as np
from data import action4_modular_parameters as params
from data import action4_modular_assembly as assembly
from data.action4_global_environment import digest
ROOT=params.ROOT
CONFIG=ROOT/'configs/data/action4_modular_layout_proposals_v001_candidate.json'
SCHEMA='action4.modular_layout_proposal.v1'
SCHEMA_V2='action4.modular_layout_proposal.v2'
STATUS_V2='LAYOUT_ORIENTATION_CONDITIONED_REQUIRES_ACTUAL_GEOMETRY'
STATUS='LAYOUT_CONDITIONED_PREVIEW_CANDIDATE'

class LayoutProposalError(ValueError):
 def __init__(self,report):self.report=report;super().__init__(json.dumps(report,allow_nan=False))

def ref(path):
 p=params.bounded(path);return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def load_config(path=CONFIG):
 c=json.loads(params.bounded(path).read_text());validate_config(c);params.verified_bytes(c['parameter_config_ref']);params.verified_bytes(c['audit_ref']);return c

def validate_config(c):
 if set(c)-({'orientation_evidence_ref','orientation_selection'} if c.get('schema')=='action4.modular_layout_proposal_config.v2' else set())!={'schema','version','status','dataset_generation_allowed','parameter_config_ref','audit_ref','search_domain','image_target_perturbation','orientation_templates','acceptance','no_formal_coverage','no_actual_render_or_pose_GT_claim'}:raise ValueError('Unknown layout overrides forbidden')
 if c['acceptance']['bbox_guard']!='usable_reference_bounds_proposal':raise ValueError('Explicit diagnostic image guard')
 if c['schema']=='action4.modular_layout_proposal_config.v1' and c['orientation_templates'].get('cross_side_fallback_only_after_all_same_side_fail') is not True:raise ValueError('Engineering cross-side fallback cannot replace valid same-side proposals')
 if c['schema'] not in ('action4.modular_layout_proposal_config.v1','action4.modular_layout_proposal_config.v2') or c['status']!='engineering_layout_preview_only' or c['dataset_generation_allowed'] is not False or c['no_formal_coverage'] is not True:raise ValueError('Explicit layout preview contract required')
 d=c['search_domain'];lo,hi=d['C_translation_z_m']
 if not .15<=lo<hi<=1.2 or d['XY_from_report_guard'] is not True or d['near_plane_m']!=.001 or not 1<=d['max_iterations_z']<=64 or not 1<=d['max_iterations_xy']<=64 or not 1<=d['max_orientation_proposals_per_cluster']<=12:raise ValueError('Finite declared translation search domain only')
 if c['image_target_perturbation']['center_each_axis_normalized']!=.01 or c['image_target_perturbation']['bbox_linear_fraction']!=.05:raise ValueError('One paired cluster with declared1percent/5percent neighborhood')
 t=c['orientation_templates']
 if t['left_same_side_frames']!=[0] or t['right_same_side_frames']!=[0,540,960] or type(t['allow_left_cross_side_engineering']) is not bool:raise ValueError('Declared source rotation templates only')
 if (c['acceptance']['require_actor_AABB_separation'] is not (c['schema']=='action4.modular_layout_proposal_config.v1')) or not 0<c['acceptance']['center_tolerance_px']<=.01 or not 0<c['acceptance']['area_relative_tolerance']<=1e-6:raise ValueError('Explicit conservative preview acceptance')
 if c['schema']=='action4.modular_layout_proposal_config.v2':
  if c['orientation_selection']!={'source_frame_before_own_side_before_shape':True,'visible_face_gate':True,'unresolved_AABB':'requires_actual_surface_check'} or c['orientation_templates']['cross_side_fallback_only_after_all_same_side_fail'] is not False:raise ValueError('Explicit evidence-first finite diagnostic selection')
  _orientation_evidence(c)
 return c

def _orientation_evidence(c):
 evidence=json.loads(params.verified_bytes(c['orientation_evidence_ref']))
 if evidence.get('schema')!='action4.layout_visible_face_evidence.v1' or evidence.get('measured_pose_GT') is not False or evidence.get('training_accepted') is not False or evidence.get('source_frame')!=540:raise ValueError('Bound explicit face observation, not measured pose')
 for reference in evidence['refs'].values():params.verified_bytes(reference)
 expected={'face':'rear_screen','CAD_outward_normal':[0.,1.,0.],'normal_dot_toward_eye_must_be_positive':True,'source_frame_priority':540}
 if evidence.get('requirements_by_side')!={s:expected for s in ('left','right')}:raise ValueError('Supported source540 rear-screen evidence only')
 if evidence.get('left_source540_fitted_pose_exists') is not False or evidence.get('left_may_borrow_right540_only_as_engineering_orientation') is not True:raise ValueError('LEFT540 cannot claim fitted pose GT')
 return evidence

def orientation_assessment(rows,combination,c,frame):
 evidence=_orientation_evidence(c)
 if frame!=evidence['source_frame']:raise ValueError('This version is explicitly bound to observed540 only')
 result={};failures=[]
 for side in ('left','right'):
  C=assembly.rigid(rows[side]['core_C_capture']);view=-C[:3,3]/np.linalg.norm(C[:3,3]);normal=C[:3,:3]@np.array(evidence['requirements_by_side'][side]['CAD_outward_normal']);dot=float(normal@view)
  result[side]={'rear_normal_dot_toward_eye':dot,'rear_face_sign_pass':bool(dot>0),'source_frame_match':combination[side]['source_frame']==frame,'orientation_is_engineering_cross_side':combination[side]['engineering_cross_side'],'not_measured_pose_GT':True}
  if dot<=0:failures.append(side+':observed_rear_face_sign_mismatch')
 return {'by_side':result,'source_frame_mismatch_count':sum(not r['source_frame_match'] for r in result.values()),'engineering_cross_side_count':sum(r['orientation_is_engineering_cross_side'] for r in result.values()),'evidence_ref':c['orientation_evidence_ref'],'not_actual_surface_or_visibility_acceptance':True},failures

def _derived_replay_tolerance(path):
 """Named computed leaves only; source values/refs, selection and flags are exact.
 Absolute tolerances match existing layout guards: metres/matrix1e-9,
 native pixels1e-6, normalized ratios1e-10. Dot/shape arithmetic uses1e-12.
 No relative tolerance, float-wide fallback, rounding or source normalization.
 """
 side='(?:left|right)';attempt=r'(?:0|[1-9][0-9]*)';xyz='[0-2]'
 orientation=r'layout_conditioning/(?:attempts/'+attempt+r'/)?orientation_evidence_assessment/by_side/'+side+r'/rear_normal_dot_toward_eye'
 if re.fullmatch(orientation,path):return 1e-12
 if re.fullmatch(r'sides/'+side+r'/core_C_capture/'+xyz+r'/3',path):return 1e-9
 if re.fullmatch(r'sides/'+side+r'/(?:brace_B_capture|whole_actor_delta_capture)/'+xyz+r'/[0-3]',path):return 1e-9
 if re.fullmatch(r'sides/'+side+r'/translation_capture_m/'+xyz,path):return 1e-9
 metric=r'derived/by_side/'+side+'/'
 if re.fullmatch(metric+r'canonical8_capture_m/[0-7]/'+xyz,path):return 1e-9
 if re.fullmatch(metric+r'(?:canonical_center_distance_to_capture_eye_m|canonical_center_optical_depth_m)',path):return 1e-9
 if re.fullmatch(metric+r'(?:canonical8_native_px/[0-7]/[0-1]|bbox_native_xyxy/[0-3])',path):return 1e-6
 if re.fullmatch(metric+r'(?:bbox_width_over_W|bbox_height_over_H|bbox_area_over_image|bbox_inside_image_fraction|bbox_center_normalized/[0-1])',path):return 1e-10
 if path=='derived/canonical_center_pair_distance_capture_m':return 1e-9
 if path=='derived/paired_horizontal_center_separation_fraction':return 1e-10
 if re.fullmatch(r'derived/reference_image_layout_diagnostic/actual_canonical8_bbox_metrics/'+side+r'/(?:width|height|area|center_x|center_y)',path):return 1e-10
 solver=r'layout_conditioning/(?:attempts/'+attempt+r'/)?side_solvers/'+side+'/'
 if re.fullmatch(solver+r'(?:final_XY_brackets_m/[0-1]/[0-1]|solved_C_translation_m/'+xyz+r')',path):return 1e-9
 if re.fullmatch(solver+r'(?:actual_area|area_relative_error)',path):return 1e-10
 if re.fullmatch(solver+r'center_error_px',path):return 1e-6
 if re.fullmatch(r'layout_conditioning/attempts/'+attempt+r'/shape_match_score',path):return 1e-12
 prior=r'(?:safety_prior|layout_conditioning/attempts/'+attempt+r'/cross_actor_prior)/'
 pieces=r'(?:core|skin|band|quick_release|usb_connector|white_module|ball|link|cable)'
 if re.fullmatch(prior+r'actor_AABB_by_side_capture_m/'+side+r'/(?:min_m|max_m)/'+xyz,path):return 1e-9
 if re.fullmatch(prior+r'piece_AABB_by_side_capture_m/'+side+'/'+pieces+r'/(?:min_m|max_m)/'+xyz,path):return 1e-9
 if re.fullmatch(prior+r'pair/(?:actor_AABB_distance_lower_bound_m|piece_AABB_minimum_distance_lower_bound_m)',path):return 1e-9
 return None

def _portable_replay_equal(actual,expected,path=()):
 """Strict JSON structure/types; tolerate only explicitly named finite derivatives."""
 if type(actual) is not type(expected):return False
 if isinstance(expected,dict):return set(actual)==set(expected) and all(_portable_replay_equal(actual[k],v,path+(str(k),)) for k,v in expected.items())
 if isinstance(expected,list):return len(actual)==len(expected) and all(_portable_replay_equal(a,b,path+(str(i),)) for i,(a,b) in enumerate(zip(actual,expected)))
 if type(expected) is float:
  if not math.isfinite(actual) or not math.isfinite(expected):return False
  if actual==expected:return True
  tolerance=_derived_replay_tolerance('/'.join(path))
  return tolerance is not None and abs(actual-expected)<=tolerance
 return actual==expected

def _parameter_config(c):return params.load_config(params.bounded(c['parameter_config_ref']['path']))

def _cluster(c,frame):
 if type(frame) is not int:raise ValueError('Integer audited source frame required')
 report=json.loads(params.verified_bytes(c['audit_ref']));rows=[r for r in report['rows'] if r['frame']==frame]
 if len(rows)!=1:raise ValueError('Exact audited paired source frame required')
 row=rows[0];metrics={r['side']:r for r in row['body_envelope_metrics']}
 if set(metrics)!={'L','R'} or any(r['training_GT'] is not False for r in metrics.values()):raise ValueError('Image proxies are not3D/trainingGT')
 return row,metrics,report['usable_reference_bounds_proposal']

def _targets(entry,c,pc,frame,jitter):
 row,metrics,guard=_cluster(c,frame);out={}
 for side,letter in [('left','L'),('right','R')]:
  source=metrics[letter];center=np.array(source['center_norm'],float);factor=1.
  if jitter:
   center += np.array([2*params._random(entry,pc,'layout:'+str(frame)+':'+side+':center'+str(i))-1 for i in [0,1]])*.01
   factor=.95+.10*params._random(entry,pc,'layout:'+str(frame)+':'+side+':linear_bbox')
  out[side]={'center_norm':center.tolist(),'bbox_area_fraction':float(source['bbox_area_fraction']*factor*factor),'bbox_linear_factor_image_target_only':factor,'observed_width_fraction':source['bbox_width_fraction'],'observed_height_fraction':source['bbox_height_fraction'],'source_frame':frame,'heuristic_edge_uncertainty_demo_px':source['heuristic_edge_uncertainty_demo_px'],'training_GT':False}
 return out,guard

def _at_depth(R,z,center,K,Q,guard,letter,iterations,near):
 rotated=np.asarray(Q)@R.T;depth=rotated[:,2]+z
 if np.any(depth<=near):raise ValueError('canonical8_near_plane_failure')
 t=np.array([0.,0.,z]);targets=np.asarray(center)*[3248.,2464.];brackets=[]
 for axis,suffix,extent in [(0,'x',3248.),(1,'y',2464.)]:
  bounds=guard[letter+'_center_'+suffix]
  if not bounds[0]<=center[axis]<=bounds[1]:raise ValueError('requested_center_outside_audited_diagnostic_layout_guard')
  lower=float(np.min((bounds[0]*extent-K[axis,2])*depth/K[axis,axis]-rotated[:,axis]));upper=float(np.max((bounds[1]*extent-K[axis,2])*depth/K[axis,axis]-rotated[:,axis]));brackets.append([lower,upper])
  for _ in range(iterations):
   mid=(lower+upper)/2;u=K[axis,axis]*(rotated[:,axis]+mid)/depth+K[axis,2];value=(u.min()+u.max())/2
   if value<targets[axis]:lower=mid
   else:upper=mid
  t[axis]=(lower+upper)/2
 C=np.eye(4);C[:3,:3]=R;C[:3,3]=t;metrics=params.project_canonical(C,K,Q,near)
 return C,metrics,brackets

def solve_translation(R,target,K,Q,guard,letter,domain):
 """Bounded scalar depth search; XY bisections solve actual bbox centers."""
 lo,hi=domain['C_translation_z_m'];args=(target['center_norm'],K,Q,guard,letter,domain['max_iterations_xy'],domain['near_plane_m']);Clo,mlo,_=_at_depth(R,lo,*args);Chi,mhi,_=_at_depth(R,hi,*args);target_area=target['bbox_area_fraction'];alo,ahi=mlo['bbox_area_over_image'],mhi['bbox_area_over_image']
 if not alo>=target_area>=ahi or alo<=ahi:raise ValueError('target_area_unreachable_in_declared_depth_domain')
 for _ in range(domain['max_iterations_z']):
  z=(lo+hi)/2;C,m,brackets=_at_depth(R,z,*args);area=m['bbox_area_over_image']
  if area>alo+1e-10 or area<ahi-1e-10:raise ValueError('nonmonotone_area_domain_requires_explicit_review')
  if area>target_area:lo,alo=z,area
  else:hi,ahi=z,area
 C,m,brackets=_at_depth(R,(lo+hi)/2,*args)
 return C,m,{'depth_iterations':domain['max_iterations_z'],'XY_iterations_each':domain['max_iterations_xy'],'final_XY_brackets_m':brackets,'Cz_domain_m':domain['C_translation_z_m'],'solved_C_translation_m':C[:3,3].tolist(),'target_area':target_area,'actual_area':m['bbox_area_over_image'],'area_relative_error':abs(m['bbox_area_over_image']-target_area)/target_area,'center_error_px':float(np.max(abs(np.array(m['bbox_center_normalized'])-target['center_norm'])*[3248,2464]))}

def _templates(c,source):
 left=[{'source_side':'left','source_frame':0,'engineering_cross_side':False}];right=[{'source_side':'right','source_frame':i,'engineering_cross_side':False} for i in [0,540,960]]
 if c['orientation_templates']['allow_left_cross_side_engineering']:left += [{'source_side':'right','source_frame':i,'engineering_cross_side':True} for i in [0,540,960]]
 combinations=[{'left':l,'right':r} for l in left for r in right]
 if len(combinations)>c['search_domain']['max_orientation_proposals_per_cluster']:raise ValueError('Explicit orientation proposal budget exhausted before sampling')
 return combinations

@lru_cache(maxsize=4)
def _local_piece_cache(path,sha256,W_json,B_json):
    archive=np.load(io.BytesIO(params.verified_bytes({'path':path,'sha256':sha256})),allow_pickle=False);W=np.array(json.loads(W_json));Bs=json.loads(B_json);out={}
    for side in ['left','right']:
        inv=np.linalg.inv(W@np.array(Bs[side]));out[side]={key[len(side)+1:-len('_vertices_m')]:assembly.transform(inv,archive[key]) for key in archive.files if key.startswith(side+'_') and key.endswith('_vertices_m')}
        if len(out[side])!=9:raise ValueError('Actual complete9piece actor cache required')
    return out


def _pair_prior(rows,pc):
    W=pc['fixed']['W_from_capture'];Bs={s:(np.array(pc['fixed']['sides'][s]['anchor_C_capture_by_frame']['0'])@np.linalg.inv(pc['fixed']['sides'][s]['A_brace_from_CAD'])).tolist() for s in ['left','right']};ref0=pc['source_refs']['actual_actor_geometry'];local=_local_piece_cache(ref0['path'],ref0['sha256'],json.dumps(W),json.dumps(Bs));bounds={};actor={}
    for side,row in rows.items():
        transformed={key:assembly.transform(np.array(row['brace_B_capture']),V) for key,V in local[side].items()};bounds[side]={key:{'min_m':V.min(0).tolist(),'max_m':V.max(0).tolist()} for key,V in transformed.items()};allV=np.vstack(list(transformed.values()));actor[side]={'min_m':allV.min(0).tolist(),'max_m':allV.max(0).tolist()}
    uncertain=[];minimum=None
    for a,ba in bounds['left'].items():
        for b,bb in bounds['right'].items():
            gap=np.maximum(np.maximum(np.array(ba['min_m'])-bb['max_m'],np.array(bb['min_m'])-ba['max_m']),0);distance=float(np.linalg.norm(gap));minimum=distance if minimum is None else min(minimum,distance)
            if not np.any(gap>0):uncertain.append({'left_piece':a,'right_piece':b,'status':'overlap_AABB_requires_actual_surface_check'})
    lower=np.maximum(actor['left']['min_m'],actor['right']['min_m']);upper=np.minimum(actor['left']['max_m'],actor['right']['max_m']);gaps=np.maximum(lower-upper,0)
    return {'actor_AABB_by_side_capture_m':actor,'piece_AABB_by_side_capture_m':bounds,'pair':{'actor_AABB_strictly_disjoint_prior':bool(np.any(gaps>0)),'actor_AABB_distance_lower_bound_m':float(np.linalg.norm(gaps)),'piece_AABB_checked_pairs':81,'piece_AABB_unresolved_count':len(uncertain),'piece_AABB_unresolved_pairs':uncertain,'piece_AABB_minimum_distance_lower_bound_m':minimum,'minimum_surface_distance_m':None},'requires_actual_geometry_contact_containment_and_masks':True,'near_plane_pass':all(bounds['min_m'][2]>pc['near_plane_m'] for bounds in actor.values())}


def _assess(metrics,targets,guard,c):
 failures=[]
 for side in ['left','right']:
  row=metrics[side];target=targets[side]
  if not row['valid_positive_depth']:failures.append(side+':near_plane')
  for field,key in [('bbox_width_over_W','width_fraction'),('bbox_height_over_H','height_fraction')]:
   if not guard[key][0]<=row[field]<=guard[key][1]:failures.append(side+':outside_'+key)
  if np.max(abs(np.array(row['bbox_center_normalized'])-target['center_norm'])*[3248,2464])>c['acceptance']['center_tolerance_px']:failures.append(side+':center_residual')
  if abs(row['bbox_area_over_image']-target['bbox_area_fraction'])/target['bbox_area_fraction']>c['acceptance']['area_relative_tolerance']:failures.append(side+':area_residual')
 separation=abs(metrics['right']['bbox_center_normalized'][0]-metrics['left']['bbox_center_normalized'][0])
 if not guard['horizontal_center_separation'][0]<=separation<=guard['horizontal_center_separation'][1]:failures.append('outside_paired_horizontal_layout_guard')
 return failures

def _environment_prior_failures(metrics,pc,family):
    failures=[];profile=pc['family_profiles'][family]
    if profile['wall_visible']:
        wall=pc['fixed']['baseline_environment']['wall_front_capture_z_m']
        for side,row in metrics.items():
            if np.max(np.asarray(row['canonical8_capture_m'])[:,2])>=wall:failures.append(side+':canonical_body_reaches_visible_wall_plane')
    return failures


def propose_layout(entry,config,source_context,source_frame=None,jitter=True):
 c=load_config(config) if not isinstance(config,dict) else validate_config(config);cfgpath=params.bounded(config) if not isinstance(config,dict) else CONFIG
 if digest(json.loads(cfgpath.read_text()))!=digest(c):raise ValueError('Explicit saved layout config binding required')
 pc=_parameter_config(c);params.validate_source_context(pc,source_context)
 if entry['target_presence']!=[True,True]:raise LayoutProposalError({'status':'LAYOUT_REQUIRES_PAIRED_TARGET_ENTRY','entry_index':entry['index']})
 regression=deepcopy(pc);regression['sampling_mode']='baseline_regression';base=params.sample_parameters(entry,regression,source_context)
 audit=json.loads(params.verified_bytes(c['audit_ref']));frame=source_frame if source_frame is not None else params._choice(entry,pc,'layout_source_cluster',audit['frames_used']);targets,guard=_targets(entry,c,pc,frame,jitter);K=np.array(source_context['K_native']);Q=pc['fixed']['canonical8_m'];attempts=[];valid=[]
 explicit_orientation=c['schema']=='action4.modular_layout_proposal_config.v2'
 for combination in _templates(c,source_context):
  if not explicit_orientation and combination['left']['engineering_cross_side'] and any(not v[5]['left']['engineering_cross_side'] for v in valid):break
  rows=deepcopy(base['sides']);metrics={};solvers={};failure=[];orientation=None
  try:
   for side,letter in [('left','L'),('right','R')]:
    template=combination[side];R=np.array(source_context['sides'][template['source_side']]['anchor_C_capture_by_frame'][str(template['source_frame'])])[:3,:3];C,metric,solver=solve_translation(R,targets[side],K,Q,guard,letter,c['search_domain']);A=np.array(source_context['sides'][side]['A_brace_from_CAD']);B=C@np.linalg.inv(A);B0=np.array(source_context['sides'][side]['base_B_capture']);rows[side].update(core_C_capture=C.tolist(),brace_B_capture=B.tolist(),whole_actor_delta_capture=(B@np.linalg.inv(B0)).tolist(),translation_capture_m=(B[:3,3]-B0[:3,3]).tolist(),rotation_brace_local_xyz_deg=[0.,0.,0.]);metrics[side]=metric;solvers[side]=solver
   failure=_assess(metrics,targets,guard,c);failure += _environment_prior_failures(metrics,pc,entry['scene_family']);prior=_pair_prior(rows,pc)
   if not explicit_orientation and prior['pair']['piece_AABB_unresolved_count']:failure.append('cross_piece_AABB_overlap_requires_surface_review_no_prior_clearance')
   if not prior['near_plane_pass']:failure.append('full_actor_vertex_near_plane_failure')
   score=sum(abs(math.log(metrics[s]['bbox_width_over_W']/targets[s]['observed_width_fraction']))+abs(math.log(metrics[s]['bbox_height_over_H']/targets[s]['observed_height_fraction'])) for s in ['left','right'])
   if explicit_orientation:
    orientation,face_failures=orientation_assessment(rows,combination,c,frame);failure+=face_failures
    prior['unresolved_AABB_disposition']='requires_actual_surface_check'
    prior['actual_surface_check_pass']=False
    prior['containment_check_pass']=False
  except ValueError as error:failure.append(str(error));prior=None;score=None
  record={'orientation_templates_by_side':combination,'failure_reasons':failure,'side_solvers':solvers,'cross_actor_prior':prior,'shape_match_score':score}
  if explicit_orientation:record['orientation_evidence_assessment']=orientation
  attempts.append(record)
  if not failure:valid.append((score,rows,metrics,solvers,prior,combination))
 if not valid:raise LayoutProposalError({'status':'NO_LAYOUT_CANDIDATE_WITHIN_FINITE_CONTRACT','source_cluster_frame':frame,'entry':entry,'attempts':attempts,'no_automatic_retry_or_quota_change':True})
 own=[v for v in valid if not v[5]['left']['engineering_cross_side']]
 if explicit_orientation:
  def ranking(item):
   assessment,_=orientation_assessment(item[1],item[5],c,frame)
   return (assessment['source_frame_mismatch_count'],assessment['engineering_cross_side_count'],item[0])
  chosen=min(valid,key=ranking)
 else:chosen=min(own or valid,key=lambda item:item[0])
 _,rows,metrics,solvers,prior,combination=chosen;p=deepcopy(base);p.update(schema=SCHEMA_V2 if explicit_orientation else SCHEMA,status=STATUS_V2 if explicit_orientation else STATUS,mode='layout_conditioned_preview',sides=rows,source_anchor_frames_by_side={s:(0 if combination[s]['engineering_cross_side'] else combination[s]['source_frame']) for s in ['left','right']})
 p['augmentation'].update(reference_or_augmentation='layout_conditioned_engineering_preview',primary_control='capture_translation_depth_conditioned_on_paired_image_cluster',counts_as_formal_coverage=False,reference_bounds_known=False)
 p['augmentation'].pop('baseline_regression_does_not_fill1800_reference_quota',None);p['augmentation'].update(one_primary_control_only=False,conditional_paired_capture_translation_coordinates=True,old4mm_small_augmentation_mode=False)
 p['derived'].update(by_side=metrics,same_source_frame_pair_anchor=False,mixed_frame_pair=(combination['left']['source_frame']!=combination['right']['source_frame'] or combination['left']['engineering_cross_side']),orientation_template_combination_is_not_same_real_frame_claim=True,reference_image_layout_request={'source_cluster_frame':frame,'targets_by_side':targets,'not_3D_GT':True})
 p['derived']['paired_horizontal_center_separation_fraction']=abs(metrics['right']['bbox_center_normalized'][0]-metrics['left']['bbox_center_normalized'][0]);p['derived']['canonical_center_pair_distance_capture_m']=float(np.linalg.norm(np.mean(metrics['left']['canonical8_capture_m'],axis=0)-np.mean(metrics['right']['canonical8_capture_m'],axis=0)))
 p['derived']['reference_image_layout_diagnostic']=params.check_reference_layout(p,params.reference_layout_constraints(pc))
 p['safety_prior']=prior;p['layout_conditioning']={'config_ref':ref(cfgpath),'config_sha256':digest(c),'audit_ref':c['audit_ref'],'source_cluster_frame':frame,'jitter':bool(jitter),'targets_by_side':targets,'orientation_templates_by_side':combination,'side_solvers':solvers,'search_domain':deepcopy(c['search_domain']),'attempts':attempts,'delta_reference':'verified_f0_base_B','orientation_source_is_not_target_video_pose_GT':True,'geometry_size_A_S_F_K_W_unchanged':True,'same_real_frame_3D_pair_claim':False,'actual_scene_and_family_effect_validation_pending':True}
 if explicit_orientation:
  p['layout_conditioning']['orientation_evidence_assessment']=orientation_assessment(rows,combination,c,frame)[0]
  p['layout_conditioning']['selection_policy']=deepcopy(c['orientation_selection'])
  p['provenance']['refs']['orientation_evidence']=c['orientation_evidence_ref']
 p['provenance']['refs'].update(layout_config=ref(cfgpath),layout_module=ref(__file__));p['provenance']['layout_config_sha256']=digest(c);json.dumps(p,allow_nan=False);return p

def validate_layout_parameters(p,context):
 """Independent consumer closure guard for the explicit larger-translation mode."""
 if (p.get('schema'),p.get('status')) not in ((SCHEMA,STATUS),(SCHEMA_V2,STATUS_V2)) or p.get('mode')!='layout_conditioned_preview' or p['augmentation']['counts_as_formal_coverage'] is not False:raise ValueError('Explicit nonproduction layout proposal mode')
 condition=p['layout_conditioning'];configref=condition['config_ref'];params.verified_bytes(configref);c=load_config(params.bounded(configref['path']));pc=_parameter_config(c);source=context['source_context'];params.validate_source_context(pc,source)
 if digest(c)!=condition['config_sha256'] or condition['audit_ref']!=c['audit_ref'] or p['target_presence']!=[True,True] or condition['same_real_frame_3D_pair_claim'] is not False:raise ValueError('Bound paired layout contract')
 if digest(p['W_from_capture'])!=digest(source['W_from_capture']) or digest(p['K_native'])!=digest(source['K_native']) or p['provenance']['source_context_sha256']!=context['source_context_sha256']:raise ValueError('Frozen W/K/source context')
 if p['entry']!=params._entries(pc).get(p['entry']['index']):raise ValueError('Immutable original plan identity')
 if p['derived']['same_source_frame_pair_anchor'] is not False or p['derived'].get('orientation_template_combination_is_not_same_real_frame_claim') is not True:raise ValueError('Synthetic layout cannot claim a real paired3D frame')
 targets,guard=_targets(p['entry'],c,pc,condition['source_cluster_frame'],condition['jitter'])
 if digest(targets)!=digest(condition['targets_by_side']):raise ValueError('Paired target perturbation changed')
 if condition['orientation_templates_by_side'] not in _templates(c,source):raise ValueError('Unapproved orientation template or cross-side claim')
 expected_anchors={side:(0 if row['engineering_cross_side'] else row['source_frame']) for side,row in condition['orientation_templates_by_side'].items()}
 if p['source_anchor_frames_by_side']!=expected_anchors:raise ValueError('Orientation template source-frame metadata changed')
 if p['provenance']['refs'].get('layout_config')!=configref or p['provenance']['refs'].get('layout_module')!=ref(__file__):raise ValueError('Layout module/config provenance binding')
 metrics={}
 for side,letter in [('left','L'),('right','R')]:
  row=p['sides'][side]
  if any(digest(row[field])!=digest(source['sides'][side][field]) for field in ['A_brace_from_CAD','S_skin_relative_to_brace','F_body_from_brace']):raise ValueError('Frozen A/S/F unchanged')
  C=assembly.rigid(row['core_C_capture']);B=assembly.rigid(row['brace_B_capture']);A=assembly.rigid(source['sides'][side]['A_brace_from_CAD']);B0=assembly.rigid(source['sides'][side]['base_B_capture']);template=condition['orientation_templates_by_side'][side];R=np.array(source['sides'][template['source_side']]['anchor_C_capture_by_frame'][str(template['source_frame'])])[:3,:3]
  if np.max(abs(C[:3,:3]-R))>1e-12 or np.max(abs(C-B@A))>1e-9 or np.any(np.asarray(row['rotation_brace_local_xyz_deg'])!=0):raise ValueError('Frozen source R/A only; no free orientation or scale')
  if not c['search_domain']['C_translation_z_m'][0]<=C[2,3]<=c['search_domain']['C_translation_z_m'][1]:raise ValueError('Declared scalar depth bound')
  if np.max(abs(np.array(row['whole_actor_delta_capture'])-B@np.linalg.inv(B0)))>1e-9 or np.max(abs(np.array(row['translation_capture_m'])-(B[:3,3]-B0[:3,3])))>1e-9:raise ValueError('Large conditional translation must be explicit, not old4mm metadata')
  metric=params.project_canonical(C,source['K_native'],pc['fixed']['canonical8_m'],c['search_domain']['near_plane_m']);metrics[side]=metric
  _,_,brackets=_at_depth(R,C[2,3],targets[side]['center_norm'],np.array(source['K_native']),pc['fixed']['canonical8_m'],guard,letter,c['search_domain']['max_iterations_xy'],c['search_domain']['near_plane_m'])
  if any(not bracket[0]-1e-12<=C[i,3]<=bracket[1]+1e-12 for i,bracket in enumerate(brackets)):raise ValueError('Image-reachable XY bracket')
  claimed=p['derived']['by_side'][side]
  if np.max(abs(np.array(metric['canonical8_native_px'])-np.array(claimed['canonical8_native_px'])))>1e-6 or any(abs(metric[k]-claimed[k])>1e-10 for k in ['bbox_width_over_W','bbox_height_over_H','bbox_area_over_image']):raise ValueError('Derived layout must come from original3D8 projection')
 expected_environment={'area_light_gain':1.,'exposure_offset_ev':0.,'wall_visible':pc['family_profiles'][p['entry']['scene_family']]['wall_visible'],'table_visible':pc['family_profiles'][p['entry']['scene_family']]['table_visible']}
 if p['environment_controls']!=expected_environment:raise ValueError('Layout-only mode preserves declared baseline environment controls')
 failure=_assess(metrics,targets,guard,c);failure += _environment_prior_failures(metrics,pc,p['entry']['scene_family'])
 if failure:raise ValueError('Conditional projection closure '+str(failure))
 expected_horizontal=abs(metrics['right']['bbox_center_normalized'][0]-metrics['left']['bbox_center_normalized'][0]);expected_distance=float(np.linalg.norm(np.mean(metrics['left']['canonical8_capture_m'],axis=0)-np.mean(metrics['right']['canonical8_capture_m'],axis=0)))
 if abs(p['derived']['paired_horizontal_center_separation_fraction']-expected_horizontal)>1e-10 or abs(p['derived']['canonical_center_pair_distance_capture_m']-expected_distance)>1e-9:raise ValueError('Derived pair spacing must be recomputed')
 if p['derived']['reference_image_layout_diagnostic']!=params.check_reference_layout(p,params.reference_layout_constraints(pc)):raise ValueError('Stale base layout diagnostic forbidden')
 prior=_pair_prior(p['sides'],pc)
 if c['schema']=='action4.modular_layout_proposal_config.v1' and prior['pair']['piece_AABB_unresolved_count']:raise ValueError('Cross-side conservative piece prior requires review')
 if not prior['near_plane_pass']:raise ValueError('Full actor vertex near-plane failure')
 explicit_orientation=c['schema']=='action4.modular_layout_proposal_config.v2'
 if (p['schema']==SCHEMA_V2)!=explicit_orientation:raise ValueError('Explicit matching layout/config schema required')
 if explicit_orientation:
  evidence,failures=orientation_assessment(p['sides'],condition['orientation_templates_by_side'],c,condition['source_cluster_frame'])
  if failures or not _portable_replay_equal(evidence,condition.get('orientation_evidence_assessment'),('layout_conditioning','orientation_evidence_assessment')):raise ValueError('Observed-face evidence failed or changed')
  expected=propose_layout(p['entry'],configref['path'],source,condition['source_cluster_frame'],condition['jitter'])
  if not _portable_replay_equal(p,expected):raise ValueError('Finite evidence-first selection/provenance changed beyond explicit derived tolerances; requires explicit new candidate')
 return True
