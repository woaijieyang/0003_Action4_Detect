"""Explicit diagnostic-only consumer of source-bound near/trunc requests.
Import/prepare do not create scenes. This cannot clear pending or accept data.
"""
from copy import deepcopy
import json
import numpy as np
from data import action4_modular_parameters as params
from data import action4_modular_layout_proposals as layouts
from data import action4_modular_geometric_effects_v002 as effects
from data import action4_modular_geometric_effects as base
from data import action4_global_environment as env
FAMILIES=('near_double','wall_transition_truncation')
SCHEMA='action4.modular_geometric_execution.v1'


def prepare(request_ref,context,*,entry_source_ref=None,entry_index=None,parent_layout_ref=None):
    """Validate original layout, then independently compose whole-actor motion."""
    from data import action4_modular_scene_sampling as sampling
    request=json.loads(params.verified_bytes(request_ref))
    out=prepare_request(request,request_ref,context,sampling.validate_parameters)
    supplied=(entry_source_ref,entry_index,parent_layout_ref)
    if any(x is not None for x in supplied):
        if any(x is None for x in supplied):raise ValueError('Complete frozen entry and separate parent-layout refs required')
        parent=json.loads(params.verified_bytes(parent_layout_ref))
        if env.digest(parent)!=env.digest(request['input_layout_proposal']):raise ValueError('Parent layout differs from source-bound effect request')
        source=json.loads(params.verified_bytes(entry_source_ref));entries=source.get('entries')
        if source.get('schema')!='action4.modular_entry_authority.v1' or source.get('status')!='FROZEN_ENTRY_IDENTITIES_ONLY_NOT_RENDER_PLAN' or source.get('dataset_generation_allowed') is not False or source.get('training_accepted') is not False:raise ValueError('Explicit nonaccepted entry authority required')
        if not isinstance(entries,list) or not entries or source.get('count')!=len(entries) or type(entry_index) is not int or any(set(e)!=params.ENTRY_KEYS for e in entries):raise ValueError('Frozen entries source and exact index required')
        if len({e['index'] for e in entries})!=len(entries) or len({e['sample_id'] for e in entries})!=len(entries):raise ValueError('Unique immutable source entry IDs')
        found=[e for e in entries if e['index']==entry_index]
        if len(found)!=1:raise ValueError('Exact entry lookup failed')
        entry=found[0]
        if set(entry)!=params.ENTRY_KEYS or entry['scene_family']!=request['family'] or entry['target_presence']!=[True,True] or entry['target_count']!=2:raise ValueError('Bound paired near/trunc family entry required')
        from data import action4_modular_synthesis_backend as backend
        backend.entry_identity(entry)
        out['entry']=deepcopy(entry);out['scene_family_controls']['family']=entry['scene_family']
        out['geometric_effect_execution'].update(entry_source_ref=deepcopy(entry_source_ref),entry_index=entry_index,parent_layout_ref=deepcopy(parent_layout_ref),entry_bound_before_render=True)
    return out


def validate_execution(parameters,context):
    if parameters.get('schema')!=SCHEMA:raise ValueError('Explicit geometric execution schema')
    info=parameters['geometric_effect_execution'];kwargs={k:info[k] for k in ('entry_source_ref','entry_index','parent_layout_ref') if k in info}
    expected=prepare(info['request_ref'],context,**kwargs)
    if env.digest(parameters)!=env.digest(expected):raise ValueError('Saved execution differs from bound request/entry/source')
    return deepcopy(parameters)


def prepare_request(request,request_ref,context,validate_upstream):
    """Dependency-injected validator is for focused CPU tests; use prepare publicly."""
    json.dumps(request,allow_nan=False)
    if env.digest(json.loads(params.verified_bytes(request_ref)))!=env.digest(request):
        raise ValueError('Exact saved effect request binding required')
    if request.get('family') not in FAMILIES:raise ValueError('Unsupported effect family')
    p=validate_upstream(request['input_layout_proposal'],context)
    if p['schema']!=layouts.SCHEMA:raise ValueError('Explicit independently validated layout only')
    if p['target_presence']!=[True,True]:raise ValueError('Current effects require paired original targets')
    if p['provenance']['source_context_sha256']!=context['source_context_sha256']:
        raise ValueError('Exact prepared source context binding')
    pc=layouts._parameter_config(layouts.load_config(p['layout_conditioning']['config_ref']['path']))
    Q=pc['fixed']['canonical8_m']; K=p['K_native']; wh=(3248,2464)
    if request['family']=='near_double':expected=effects.propose_near_double(p,Q,K,wh)
    else:
        c=request['effect_controls']
        expected=base.propose_truncation(p,Q,K,c['side'],c['edge'],c['outside_fraction_of_pretranslation_bbox'],wh)
    if set(request)!=set(expected):raise ValueError('Exact effect request schema')
    # Extra source-cache prior evidence is retained but never authorizes execution.
    for name in set(expected)-{'effect_evidence'}:
        if env.digest(request[name])!=env.digest(expected[name]):raise ValueError('Effect control/authority changed: '+name)
    for name,value in expected['effect_evidence'].items():
        if env.digest(request['effect_evidence'].get(name))!=env.digest(value):raise ValueError('Effect projection evidence changed: '+name)
    out=deepcopy(p);out['schema']=SCHEMA;out['status']='GEOMETRIC_EFFECT_DIAGNOSTIC_PENDING_ACTUAL_PIXELS'
    if request['family']=='near_double':deltas=request['effect_controls']['whole_actor_delta_capture_by_side']
    else:
        c=request['effect_controls'];deltas={s:(c['whole_actor_delta_capture'] if s==c['side'] else np.eye(4).tolist()) for s in ('left','right')}
    for side in ('left','right'):
        row=out['sides'][side];D=base.rigid(deltas[side]);B=D@base.rigid(p['sides'][side]['brace_B_capture']);C=D@base.rigid(p['sides'][side]['core_C_capture'])
        if np.max(abs(C-B@base.rigid(row['A_brace_from_CAD'])))>1e-9:raise ValueError('Whole actor fixed A closure')
        B0=base.rigid(context['source_context']['sides'][side]['base_B_capture'])
        row.update(core_C_capture=C.tolist(),brace_B_capture=B.tolist(),whole_actor_delta_capture=(B@np.linalg.inv(B0)).tolist(),translation_capture_m=(B[:3,3]-B0[:3,3]).tolist())
    # Original source/plan IDs and family are immutable; effect is a separate contract.
    out['derived']={'by_side':{s:params.project_canonical(out['sides'][s]['core_C_capture'],K,Q,.001) for s in ('left','right')},'original8_unclamped':True,'actual_masks_pending':True}
    out.pop('layout_conditioning');out.pop('safety_prior',None)
    out['geometric_effect_execution']={'request_ref':deepcopy(request_ref),'request_sha256':env.digest(request),'requested_family':request['family'],'upstream_layout_sha256':env.digest(p),'upstream_entry':deepcopy(p['entry']),'source_context_sha256':context['source_context_sha256'],'consumer_ref':layouts.ref(__file__),'effect_module_refs':{'base':layouts.ref(base.__file__),'v002':layouts.ref(effects.__file__)},'clears_family_pending':False,'post_render_effect_verification_required':True,'production_gate_required':True,'dataset_generation_allowed':False,'training_accepted':False}
    json.dumps(out,allow_nan=False)
    return out


def evaluated_exports(session):
    """Use original canonical exporter and evaluated actual core/camera matrices."""
    from data import action4_modular_pixels as pixels
    from data import action4_canonical_export as canonical
    from data import action4_modular_render as camera
    session.verifier(session.scene);scene=session.scene;out={};K=camera.actual_K(scene)
    if (scene.render.resolution_x,scene.render.resolution_y,scene.render.resolution_percentage)!=(3248,2464,100):raise ValueError('Frozen native frame required')
    if not np.allclose(K,session.parameters['K_native'],atol=1e-6,rtol=0):raise ValueError('Actual K changed')
    for instance,target in session.targets().items():
        out[instance]=canonical.build_export(canonical=session.contract['canonical_definition'],instance_id=instance,object_to_world_m=pixels._evaluated_world_matrix(target['core']),camera_to_world_blender_m=[list(r) for r in scene.camera.matrix_world],K=K,image_size_wh=[3248,2464],near_plane_m=scene.camera.data.clip_start)
        C=np.asarray(out[instance]['object_to_capture_cv_m']);side='left' if instance==1 else 'right'
        if not np.allclose(C,session.parameters['sides'][side]['core_C_capture'],atol=2e-6,rtol=0):raise ValueError('Actual evaluated core differs from full actor requested C')
    return out


def verify_actual_geometry(session):
    """Pre-render geometry diagnostic; mask effect and acceptance remain pending."""
    from data import action4_modular_assembly as assembly
    exports=evaluated_exports(session);report=session.contract['actor_collision_diagnostics']
    if set(exports)!={1,2} or report['outside_or_unresolved_surface_pairs']!=0:raise ValueError('Two actual targets and existing collision checks required')
    capture=np.linalg.inv(base.rigid(session.parameters['W_from_capture']));near=session.scene.camera.data.clip_start
    for binding in session.bindings.values():
        for name in binding['names']:
            obj=session.scene.objects[name]
            if obj.type=='MESH' and not obj.hide_render:
                V,_=assembly.world_geometry(obj)
                if np.min(base.transform(capture,V)[:,2])<=near:raise ValueError('Complete actor near-plane failure')
    info=session.parameters['geometric_effect_execution'];family=info['requested_family']
    if family=='near_double':
        views={}
        for i,side in ((1,'left'),(2,'right')):
            points=np.asarray(exports[i]['points_native_px'],float);lo=points.min(0);hi=points.max(0)
            views[side]={'bbox_xyxy':[*lo,*hi],'center_norm':((lo+hi)/2/[3248,2464]).tolist(),'corner_inframe':exports[i]['point_inframe']}
        condition=effects.near_double_metrics(views)
        if not condition['projection_condition_pass']:raise ValueError('Actual evaluated near-double proximity failed')
    else:
        request=json.loads(params.verified_bytes(info['request_ref']));i=1 if request['effect_controls']['side']=='left' else 2
        inside=exports[i]['point_inframe']
        if all(inside) or not any(inside):raise ValueError('Actual truncation original8 partial-inframe required')
        condition={'partial_original8_inframe':True,'actual_mask_truncation_unknown':True}
    return {'status':'ACTUAL_GEOMETRY_DIAGNOSTIC_MASKS_PENDING','request_ref':info['request_ref'],'exports':exports,'condition':condition,'environment_collision_accepted':False,'containment_accepted':False,'dataset_generation_allowed':False,'training_accepted':False}


def verify_record(parameters,geometry_evidence,record,collision_report):
    """After existing render_frame; original8 and mask numbers must match actual geometry.
    This is an engineering effect check. Existing host PNG/support/production gates
    remain mandatory; no parameter or record is promoted by this function.
    """
    info=parameters['geometric_effect_execution']
    if geometry_evidence.get('status')!='ACTUAL_GEOMETRY_DIAGNOSTIC_MASKS_PENDING' or geometry_evidence.get('request_ref')!=info['request_ref']:raise ValueError('Actual bound scene geometry evidence required')
    if any(record.get(k)!=v for k,v in parameters['entry'].items()):raise ValueError('Original immutable plan entry changed')
    if record.get('width')!=3248 or record.get('height')!=2464 or record.get('excluded_instances') or record.get('target_count')!=2:raise ValueError('Two retained actual native targets required')
    rows=record['instances']
    if len(rows)!=2 or {r['instance_id'] for r in rows}!={1,2}:raise ValueError('Original physical target IDs required')
    for row in rows:
        actual=geometry_evidence['exports'].get(row['instance_id'],geometry_evidence['exports'].get(str(row['instance_id'])))
        for key in ('corner_ids','corner_order','points_object_mm','points_object_m','point_inframe','image_size_wh'):
            if row['canonical8_geometry'].get(key)!=actual[key]:raise ValueError('Actual original8 export changed: '+key)
        for key in ('points_native_px','points_capture_m','object_to_capture_cv_m','K_native'):
            value=np.asarray(row['canonical8_geometry'].get(key),float);expected=np.asarray(actual[key],float)
            if value.shape!=expected.shape or not np.isfinite(value).all() or not np.allclose(value,expected,atol=1e-5,rtol=0):raise ValueError('Actual evaluated canonical geometry changed: '+key)
        side='left' if row['instance_id']==1 else 'right'
        trusted=parameters['derived']['by_side'][side]
        for key,reference in (('points_native_px',trusted['canonical8_native_px']),('points_capture_m',trusted['canonical8_capture_m']),('object_to_capture_cv_m',parameters['sides'][side]['core_C_capture'])):
            if not np.allclose(np.asarray(row['canonical8_geometry'][key],float),reference,atol=1e-5,rtol=0):raise ValueError('Record geometry differs from source-bound executed transforms')
        numbers=[row.get(k) for k in ('area_full_amodal_px','area_inframe_amodal_px','area_visible_px')]
        if any(type(n) is not int or n<0 for n in numbers) or not numbers[0]>=numbers[1]>=numbers[2]>0:raise ValueError('Finite real mask counts required')
    if collision_report.get('outside_or_unresolved_surface_pairs')!=0:raise ValueError('Actual collision diagnostic required')
    if info['requested_family']=='near_double':result=effects.actual_near_double_check(record,collision_report)
    elif info['requested_family']=='wall_transition_truncation':
        request=json.loads(params.verified_bytes(info['request_ref']));instance=1 if request['effect_controls']['side']=='left' else 2
        result=base.mask_effect_check('truncation',record,instance)
        row=next(r for r in rows if r['instance_id']==instance)
        result['actual_mask_truncation_fraction']=1-row['area_inframe_amodal_px']/row['area_full_amodal_px']
    else:raise ValueError('Unsupported effect')
    return {**result,'status':'ENGINEERING_EFFECT_CHECK_ONLY','request_ref':info['request_ref'],'clears_family_pending':False,'production_gate_required':True,'required_remaining_gates':['host PNG/SHA validation','existing 10percent/16px support guard','environment contact and containment acceptance','independent generation release and actual128 review','all-record formal acceptance'],'dataset_generation_allowed':False,'training_accepted':False}


def validate_record_effect(parameters,record):
    """Host/release hook; compare record proof and original8 to validated execution.
    Host PNG verification in the existing pipeline remains required separately.
    """
    proof=record.get('geometric_effect_proof')
    if not isinstance(proof,dict) or set(proof)!={'geometry','collision','check'}:raise ValueError('Actual per-effect proof required before generation release')
    expected=verify_record(parameters,proof['geometry'],record,proof['collision'])
    if env.digest(expected)!=env.digest(proof['check']) or expected.get('effect_condition_pass') is not True:raise ValueError('Actual effect failed or stale per-effect proof')
    return expected
