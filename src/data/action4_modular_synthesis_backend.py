"""Explicit released-parameter backend for the existing immutable synthesis pipeline.

No planning, optimization, Blender initialization or training at import time.
Candidates remain unaccepted; independently reviewed generation releases govern
manifest promotion only after the existing complete assembly acceptance checks.
The release is a finite entry-to-parameter map, never a family-label substitution.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from data import action4_synthesis as synth

ROOT = Path(__file__).resolve().parents[2]
ENTRY_FIELDS = ('sample_id','index','seed','scene_family','target_count','group_id',
                'background_source_id','background_seed','split','target_presence')
BACKEND = 'modular_composed'
CURRENT_PRESENCE_QUOTA_POLICY_REF = {
    'path':'configs/data/action4_pilot2000_presence_quota_v003.json',
    'sha256':'43cda95e993cf0fc4a7ee8ac958fd2167bfe25c88c1e456dd7eef0d65ec3fd05',
}
FAMILIES = {'wall_rear','table_fastening','table_turn','near_double',
            'self_forearm_occlusion','cloth_lift','wall_transition_truncation','hard_negative'}


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _ref(ref):
    if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:
        raise ValueError('Explicit path/SHA256 reference required')
    path = Path(ref['path'])
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Root-relative release paths required')
    path = synth._path(path,exists=True)
    if synth._hash(path)!=ref['sha256']:
        raise ValueError('Release source SHA mismatch: '+str(path))
    return path


def entry_identity(entry):
    result={name:deepcopy(entry[name]) for name in ENTRY_FIELDS}
    synth.target_indices(result)
    if result['split'] not in ('train','eval','test'):
        raise ValueError('Frozen split required')
    if any(type(result[k]) is not int or result[k]<0 for k in ('index','seed','background_seed')):
        raise ValueError('Frozen index/seed required')
    if result['scene_family'] not in FAMILIES:
        raise ValueError('Unknown modular family')
    return result


def _release(recipe,*,_index=None):
    if _index is not None:
        _index.require_recipe(recipe)
        if _index.release_data is not None:return _index.release_data
    read_ref=_index.read_ref if _index is not None else lambda ref:synth._read(_ref(ref))
    check_ref=_index.check_ref if _index is not None else _ref
    spec=recipe['modular_backend']
    if set(spec)!={'schema','base_kind','scene_config_ref','source_context_ref','sampling_bounds_ref','parameters_release_ref'} or spec['schema']!='action4.modular_synthesis_backend.v2' or spec['base_kind']!='palmar_wearing':
        raise ValueError('Explicit modular backend release contract required')
    scene=read_ref(spec['scene_config_ref'])
    source=read_ref(spec['source_context_ref'])
    release=read_ref(spec['parameters_release_ref'])
    if _index is not None:_index.events['full_release_scans']+=1;_index.events['release_entry_visits']+=len(release['entries'])
    check_ref(spec['sampling_bounds_ref'])
    if release.get('schema')!='action4.modular_parameter_release.v2' or release.get('source_context_sha256')!=_digest(source) or release.get('base_kind')!='palmar_wearing' or release.get('sampling_bounds_ref')!=spec['sampling_bounds_ref']:
        raise ValueError('Parameter release/source context binding')
    if source['scene_config_ref']!=spec['scene_config_ref']:
        raise ValueError('Scene configuration differs from released source context')
    if scene.get('schema')!='action4.wearing_sampling_base.v1' or scene.get('base_kind')!='palmar_wearing' or source.get('base_kind')!='palmar_wearing':
        raise ValueError('Explicit reviewed wearing sampling base required; no old-A fallback')
    if recipe['resolution_wh']!=[3248,2464]:
        raise ValueError('Frozen native resolution required; K is never rescaled here')
    if recipe['render']['samples']!=64 or recipe['render']['mask_samples']!=1:
        raise ValueError('Modular RGB64/visible1 required')
    if {f['name'] for f in recipe['families']}!=FAMILIES:
        raise ValueError('Preserve all eight families, including pending ones')
    rows={}
    for row in release['entries']:
        if set(row) not in ({'entry','parameters_ref'},{'entry','parameters_ref','pose_execution_ref'}):
            raise ValueError('Finite entry/parameter reference required')
        entry=entry_identity(row['entry']);key=entry['sample_id']
        if key in rows:raise ValueError('Duplicate released sample ID')
        parameter=read_ref(row['parameters_ref'])
        if (parameter.get('schema')=='action4.modular_pose_controls.v1')!=('pose_execution_ref' in row):raise ValueError('Explicit pose formal binding required only for pose parameters')
        if 'pose_execution_ref' in row:
            from data import action4_pose_formal_adapter as pose_adapter
            pose_adapter.validate_binding(row,_index=_index)
        rows[key]=row
    if not rows:raise ValueError('Empty parameter release')
    result=(spec,source,release,rows)
    if _index is not None:_index.release_data=result
    return result


def validate_recipe(recipe,*,_index=None):
    _release(recipe,_index=_index)
    return recipe


def asset_identities(recipe,*,_index=None):
    spec,source,release,rows=_release(recipe,_index=_index)
    check_ref=_index.check_ref if _index is not None else _ref
    read_ref=_index.read_ref if _index is not None else lambda ref:synth._read(_ref(ref))
    hash_path=_index.hash_path if _index is not None else synth._hash
    refs=dict(source['frozen_refs'])
    for ref in [spec['scene_config_ref'],spec['source_context_ref'],spec['sampling_bounds_ref'],spec['parameters_release_ref'],
                *[r['parameters_ref'] for r in rows.values()]]:
        check_ref(ref)
        if ref['path'] in refs and refs[ref['path']]!=ref['sha256']:
            raise ValueError('Conflicting release reference')
        refs[ref['path']]=ref['sha256']
    pose_context=None
    for row in rows.values():
        if 'pose_execution_ref' in row:
            from data import action4_pose_formal_adapter as pose_adapter
            from data import action4_modular_scene_sampling as sampling
            if pose_context is None:pose_context=(_index.sampling_context(spec,source,release) if _index is not None else sampling.prepare_base(_ref(spec['scene_config_ref'])))
            for path,digest in pose_adapter.dependency_refs(row,pose_context,_index=_index).items():
                if path in refs and refs[path]!=digest:raise ValueError('Conflicting pose formal source reference')
                refs[path]=digest
            refs['src/data/action4_pose_formal_adapter.py']=hash_path('src/data/action4_pose_formal_adapter.py')
        parameter=read_ref(row['parameters_ref'])
        if parameter.get('schema')=='action4.modular_geometric_execution.v1':
            info=parameter['geometric_effect_execution']
            dependencies=[info[k] for k in ('request_ref','parent_layout_ref','entry_source_ref','consumer_ref')]
            dependencies.extend(info['effect_module_refs'].values())
            for dependency in dependencies:
                check_ref(dependency)
                if dependency['path'] in refs and refs[dependency['path']]!=dependency['sha256']:raise ValueError('Conflicting geometric effect source reference')
                refs[dependency['path']]=dependency['sha256']
    for path in ('src/data/action4_modular_geometric_consumer.py',
                 'src/data/action4_modular_geometric_effects.py','src/data/action4_modular_geometric_effects_v002.py',
                 'src/data/action4_modular_synthesis_backend.py',
                 'src/data/action4_modular_parameters.py','src/data/action4_modular_layout_proposals.py',
                 'src/data/action4_dataset.py'):
        refs[path]=hash_path(path)
    for path,digest in refs.items():check_ref({'path':path,'sha256':digest})
    return refs


def validate_plan(plan,*,_index=None):
    if _index is not None:_index.require_plan(plan)
    if 'entry_authority_ref' in plan:
        authority=(_index.read_ref(plan['entry_authority_ref']) if _index is not None else synth._read(_ref(plan['entry_authority_ref'])))
        allocated=(_index.plan_authority(plan) if _index is not None else synth.build_entry_authority(plan['recipe'],plan['count'],plan['seed']))
        if authority!=allocated or authority['entries']!=plan['entries']:
            raise ValueError('Final render plan differs from immutable entry authority')
    _,_,_,rows=_release(plan['recipe'],_index=_index)
    if len(rows)!=len(plan['entries']):raise ValueError('Release must cover every planned entry exactly once')
    seen=set();groups={}
    for entry in plan['entries']:
        e=entry_identity(entry);key=e['sample_id']
        if key in seen or key not in rows or rows[key]['entry']!=e:
            raise ValueError('Plan/release entry mismatch or duplicate')
        seen.add(key)
        if 'pose_execution_ref' in rows[key]:
            from data import action4_pose_formal_adapter as pose_adapter
            pose_adapter.validate_binding(rows[key],plan=plan,_index=_index)
        split=groups.setdefault(e['group_id'],e['split'])
        if split!=e['split']:raise ValueError('Group crosses frozen splits')
    if len(seen)!=plan['count'] or plan['split_counts']!={s:sum(e['split']==s for e in plan['entries']) for s in ('train','eval','test')}:
        raise ValueError('Declared plan count/splits differ from frozen entries')
    if plan['count']==2000:
        # Same largest-remainder allocation and stable tie order as create_plan.
        # Weights are recipe authority: historical80% and current92% both retain
        # their own immutable plans; a later user quota must not be hardcoded.
        expected={}
        for split,count in (('train',1600),('eval',200),('test',200)):
            weights=synth.instance_weights_for_split(plan['recipe'],split)
            if len(weights)!=3 or not all(math.isfinite(float(w)) and float(w)>=0 for w in weights) or not any(float(w)>0 for w in weights):
                raise ValueError('Finite nonnegative per-split presence weights with positive total required')
            raw=[count*float(w)/sum(weights) for w in weights]
            quota=[math.floor(value) for value in raw]
            for k in sorted(range(3),key=lambda k:raw[k]-quota[k],reverse=True)[:count-sum(quota)]:quota[k]+=1
            expected[split]={str(k):quota[k] for k in range(3)}
        actual={s:{str(k):sum(e['split']==s and e['target_count']==k for e in plan['entries']) for k in range(3)} for s in expected}
        if plan['split_counts']!={'train':1600,'eval':200,'test':200} or actual!=expected:
            raise ValueError('Frozen recipe-derived pilot2000 split and presence quotas differ')
    if {e['scene_family'] for e in plan['entries']}!=FAMILIES:
        raise ValueError('Every frozen production/candidate plan must preserve all eight families')
    return plan


def prepare_context(recipe,*,_index=None):
    from data import action4_modular_scene_sampling as sampling
    spec,source,release,rows=_release(recipe,_index=_index)
    if _index is not None:return _index.sampling_context(spec,source,release),rows
    context=sampling.prepare_base(_ref(spec['scene_config_ref']))
    if context.get('base_kind')!='palmar_wearing' or context['source_context'].get('base_kind')!='palmar_wearing':
        raise ValueError('Prepared context is not the explicit palmar wearing factory')
    if context['source_context']!=source or context['source_context_sha256']!=release['source_context_sha256']:
        raise ValueError('Prepared context differs from released portable context')
    sampling.verify_sources(context)
    return context,rows


def parameters_for(entry,context,rows):
    from data import action4_modular_scene_sampling as sampling
    e=entry_identity(entry);row=rows.get(e['sample_id'])
    if row is None or row['entry']!=e:raise ValueError('Missing/mismatched frozen parameter entry')
    p=synth._read(_ref(row['parameters_ref']))
    if p['entry']!=e:raise ValueError('Parameter JSON differs from full planned entry')
    if p.get('schema')=='action4.modular_pose_controls.v1':
        from data import action4_pose_formal_adapter as pose_adapter
        return pose_adapter.validate_binding(row,context)[0]
    # Existing consumer rejects pending families/effects/reference bounds before scene creation.
    if p.get('schema')=='action4.modular_geometric_execution.v1':
        from data import action4_modular_geometric_consumer as geometric
        if p['geometric_effect_execution'].get('entry_bound_before_render') is not True:
            raise ValueError('Formal renderer requires frozen geometric family entry before render')
        return geometric.validate_execution(p,context)
    return sampling.validate_parameters(p,context)


def validate_record(record,entry=None,plan_sha256=None,plan=None,release_rows=None,*,validated_plan=None,run_token=None,_in_context=False):
    if validated_plan is not None and not _in_context:
        return validated_plan.validate_record(record,entry,plan_sha256,plan,release_rows,run_token)
    if validated_plan is not None:
        plan,release_rows,parameters=validated_plan.active_record_values(record,run_token)
    if plan is None and 'pose_controls_proof' in record:
        from data import action4_pose_formal_adapter as pose_adapter
        plan,release_rows=pose_adapter.resolve_record_plan(record)
    e=entry_identity(entry if entry is not None else record)
    if any(record.get(k)!=v for k,v in e.items()):raise ValueError('Record changed frozen entry')
    expected=[i for i in synth.target_indices(e) if i]
    actual=[i['instance_id'] for i in record['instances']]
    if actual!=expected or record.get('excluded_instances'):
        raise ValueError('Sparse LEFT1/RIGHT2 quota differs; excluded targets require a retained failure')
    if record.get('asset_backend')!=BACKEND or record.get('target_count')!=len(expected) or record.get('planned_target_count')!=e['target_count']:
        raise ValueError('Modular record/count identity')
    meta=record['modular_backend']
    if meta.get('base_kind')!='palmar_wearing' or meta.get('parameter_counts_as_formal_coverage') is not False:
        raise ValueError('Candidate parameter metadata must remain false on the wearing base')
    if meta['entry_sha256']!=_digest(e) or not isinstance(meta.get('parameters_sha256'),str):
        raise ValueError('Record parameter/entry provenance')
    if validated_plan is None:parameters=synth._read(_ref(meta['parameters_ref']))
    if parameters['entry']!=e or _digest(parameters)!=meta['parameters_sha256'] or parameters['provenance']['source_context_sha256']!=meta['source_context_sha256']:
        raise ValueError('Record parameter provenance differs from released JSON')
    if plan is not None:
        if record['pixel_source_identities']!=plan['asset_identities'] or meta['recipe_sha256']!=plan['recipe_sha256']:
            raise ValueError('Record source/recipe identities differ from frozen plan')
        rows=release_rows if release_rows is not None else _release(plan['recipe'])[3]
        if rows[e['sample_id']]['parameters_ref']!=meta['parameters_ref']:
            raise ValueError('Record parameters differ from frozen release')
    if plan_sha256 is not None and meta['plan_sha256']!=plan_sha256:
        raise ValueError('Record belongs to another frozen plan')
    if parameters.get('training_accepted') is not False or parameters.get('dataset_generation_allowed') is not False or parameters['augmentation'].get('counts_as_formal_coverage') is not False:
        raise ValueError('Candidate parameter JSON cannot self-promote training/formal coverage')
    if record.get('training_accepted') is not False or record.get('dataset_generation_allowed') is not False:
        raise ValueError('Child renderer cannot accept training data')
    if parameters.get('schema')=='action4.modular_pose_controls.v1':
        from data import action4_pose_formal_adapter as pose_adapter
        if plan is None:
            pose_row={'entry':e,'parameters_ref':meta['parameters_ref'],'pose_execution_ref':record['pose_controls_proof']['binding_ref']}
        else:pose_row=rows[e['sample_id']]
        pose_adapter.validate_record(pose_row,record,plan,validated_plan=validated_plan,run_token=run_token)
    if parameters.get('schema')=='action4.modular_geometric_execution.v1':
        from data import action4_modular_geometric_consumer as geometric
        if plan is not None and parameters['geometric_effect_execution'].get('entry_source_ref')!=plan.get('entry_authority_ref'):
            raise ValueError('Geometric execution source differs from final plan entry authority')
        geometric.validate_record_effect(parameters,record)
    return record


def validate_host_record(record,*,validation_session=None):
    if validation_session is not None:return validation_session.validate_host_record(record)
    from data import action4_modular_pixels as pixels
    validate_record(record)
    return pixels.validate_host_pngs(record)


def validate_formal_targets(record,entry,quality_gate):
    """Count only effective planned targets at formal promotion. The caller's
    existing validate_shard has already decoded/hash-checked the actual masks
    and verified area_visible/inframe/full against PNG pixels. Recompute support
    from those verified areas with the same original mask_measurements affine;
    never trust a producer's ignore=false or claimed full-input support alone.
    Candidate record validation deliberately remains unchanged.
    """
    from data.action4_adapter import DataAdapter
    from data.action4_dataset import is_ignored
    fraction=quality_gate['minimum_visible_fraction']
    minimum=quality_gate['minimum_visible_fullinput_area_px']
    if fraction!=.1 or minimum!=16:
        raise ValueError('The original10percent/16fullinputpixel support rules cannot change')
    expected=[i for i in synth.target_indices(entry) if i]
    if [i['instance_id'] for i in record['instances']]!=expected or record.get('excluded_instances'):
        raise ValueError('Every planned positive needs its own effective target; no quota relabeling')
    affine,_=DataAdapter.letterbox_affine((record['width'],record['height']),(1024,768))
    scale_area=float(affine[0,0]*affine[1,1]);effective=[]
    for instance in record['instances']:
        if is_ignored(instance):raise ValueError('Ignored/unobservable target cannot count in formal presence quotas')
        av,ai,af=(instance[name] for name in ('area_visible_px','area_inframe_amodal_px','area_full_amodal_px'))
        if any(type(v) is not int or v<0 for v in (av,ai,af)) or ai==0 or ai>af:
            raise ValueError('Verified positive native mask areas required for formal targets')
        support=av*scale_area
        if av/ai<fraction or support<minimum:
            raise ValueError('Actual verified mask support cannot count as an effective formal target')
        claimed=instance.get('visible_area_at_fullinput_px')
        if isinstance(claimed,bool) or not isinstance(claimed,(int,float)) or not math.isfinite(claimed) or claimed!=support:
            raise ValueError('Claimed full-input support differs from the original mask-area calculation')
        if instance.get('bbox_visible_xyxy') is None:
            raise ValueError('Effective formal target needs its actual visible mask bbox, as in Action4Dataset')
        effective.append(instance['instance_id'])
    return effective


def generation_bindings(plan,plan_ref,context):
    """Independent review binds the frozen plan/source/geometry/range, not a flag
    asserted by the candidate parameter. No review document is generated here.
    """
    _ref(plan_ref)
    _ref(CURRENT_PRESENCE_QUOTA_POLICY_REF)
    spec,source,_,_=_release(plan['recipe'])
    geometry={path:{'path':path,'sha256':digest} for path,digest in source['frozen_refs'].items()
              if Path(path).suffix.lower() in ('.npz','.blend','.stl')}
    canonical=context['wearing_context']['canonical_definition']['source_manifest']
    _ref(canonical)
    if source['frozen_refs'].get(canonical['path'])!=canonical['sha256']:
        raise ValueError('Canonical original8 source must be in the frozen closure')
    geometry[canonical['path']]=canonical
    if not any(Path(path).suffix=='.blend' for path in geometry) or not any(Path(path).suffix=='.npz' for path in geometry):
        raise ValueError('Complete actual wearing geometry cache/asset references required')
    return {'schema':'action4.modular_generation_binding.v1','base_kind':'palmar_wearing',
            'plan':plan_ref,'recipe_sha256':plan['recipe_sha256'],
            'presence_quota_policy_ref':deepcopy(CURRENT_PRESENCE_QUOTA_POLICY_REF),
            'source_context_sha256':context['source_context_sha256'],
            'asset_identities_sha256':_digest(plan['asset_identities']),
            'sampling_bounds_ref':spec['sampling_bounds_ref'],
            'parameters_release_ref':spec['parameters_release_ref'],
            'geometry_refs':geometry,'planned_entries_sha256':_digest(plan['entries'])}


def validate_presence_quota(plan,policy_ref):
    """Current formal admission is bound to root's policy; historical candidate
    plans remain valid under their own recipe but cannot satisfy this policy.
    """
    if policy_ref!=CURRENT_PRESENCE_QUOTA_POLICY_REF:
        raise ValueError('Formal generation requires the current reviewed presence quota policy')
    policy=synth._read(_ref(policy_ref))
    if policy.get('schema')!='action4.presence_quota_policy.v1':
        raise ValueError('Explicit presence quota policy schema required')
    entries=plan['entries'];total=len(entries)
    actual_total={str(k):sum(e['target_count']==k for e in entries) for k in range(3)}
    expected=policy['expected_counts_by_split']
    actual={split:{'total':sum(e['split']==split for e in entries),
                   **{str(k):sum(e['split']==split and e['target_count']==k for e in entries) for k in range(3)}} for split in expected}
    if plan['count']!=policy['total_samples'] or total!=policy['total_samples'] or actual!=expected or actual_total!=policy['expected_counts_total']:
        raise ValueError('Accepted plan differs from current presence quota policy counts')
    if actual_total['2']/total<policy['dual_fraction']:
        raise ValueError('Accepted dual-camera fraction is below the policy requirement')
    return policy_ref


def validate_generation_release(plan,plan_ref,records,acceptance):
    """Reachable promotion: candidates stay false; an independent reviewed
    generation release AND existing native/128/six-criteria checks authorize the
    assembled manifest. Every original record/parameter is checked before that.
    """
    from data import action4_modular_scene_sampling as sampling
    validate_plan(plan)
    if plan['asset_identities']!=asset_identities(plan['recipe']):
        raise ValueError('Generation source closure no longer matches the fixed plan')
    context,rows=prepare_context(plan['recipe'])
    expected=generation_bindings(plan,plan_ref,context)
    release_ref=acceptance.get('generation_release_ref')
    if release_ref is None:raise ValueError('Reviewed generation release required; candidate flags cannot promote data')
    release=synth._read(_ref(release_ref))
    if set(release)!={'schema','status','bindings','reviews','presence_quota_policy_ref'} or release['schema']!='action4.modular_generation_release.v1' or release['status']!='REVIEWED_FOR_BOUND_PLAN' or release['bindings']!=expected:
        raise ValueError('Generation release differs from fixed plan/source/geometry/sampling bounds')
    validate_presence_quota(plan,release['presence_quota_policy_ref'])
    if release['presence_quota_policy_ref']!=expected['presence_quota_policy_ref']:raise ValueError('Generation review quota policy binding differs')
    if set(release['reviews'])!={'spec','root'} or release['reviews']['spec']['path']==release['reviews']['root']['path']:
        raise ValueError('Distinct explicit SPEC and root generation reviews required')
    for role,ref in release['reviews'].items():
        review=synth._read(_ref(ref))
        if review.get('schema')!='action4.modular_generation_review.v1' or review.get('role')!=role or review.get('decision')!='approved_generation' or not review.get('reviewer') or not review.get('observations') or review.get('binding_sha256')!=_digest(expected):
            raise ValueError('Generation '+role+' review does not approve these exact bindings')
    by_id={r['sample_id']:r for r in records}
    if len(by_id)!=len(records) or set(by_id)!={e['sample_id'] for e in plan['entries']}:
        raise ValueError('Generation acceptance needs every record exactly once')
    # The actual consumer rechecks every family and bounded control. Six pending
    # effects cannot be waived by a review document or relabeled as wall_rear.
    for entry in plan['entries']:
        parameters_for(entry,context,rows)
        validate_record(by_id[entry['sample_id']],entry,plan_ref['sha256'],plan,rows)
        validate_formal_targets(by_id[entry['sample_id']],entry,plan['recipe']['quality_gate'])
    sampling.verify_sources(context)
    return release_ref


def render_entry(plan,entry,output_dir):
    from data import action4_modular_scene_sampling as sampling
    from data import action4_modular_pixels as pixels
    validate_plan(plan)
    if plan['asset_identities']!=asset_identities(plan['recipe']):raise ValueError('Frozen backend sources changed')
    context,rows=prepare_context(plan['recipe'])
    return _render_prepared_entry(plan,entry,output_dir,context,rows)


def _render_prepared_entry(plan,entry,output_dir,context,rows,*,render_scope=None):
    from data import action4_modular_scene_sampling as sampling
    from data import action4_modular_pixels as pixels
    if render_scope is not None:
        if type(render_scope) is not _RenderEntryScope:raise ValueError('Explicit live worker entry scope required')
        parameters=render_scope.parameters_for(entry,plan,context,rows)
    else:parameters=parameters_for(entry,context,rows)
    if 'pose_execution_ref' in rows[entry['sample_id']] and synth._read(_ref(rows[entry['sample_id']]['pose_execution_ref'])).get('schema')=='action4.pose_formal_binding.v2':
        return _render_composition_entry(plan,entry,output_dir,context,rows,parameters,render_scope=render_scope)
    runtime=synth._run_output_dir(output_dir)
    pose_case=parameters.get('schema')=='action4.modular_pose_controls.v1'
    source_before=deepcopy(plan['asset_identities']) if pose_case and render_scope is None else None
    session=None
    try:
        with sampling.create_scene_for_entry(context,parameters) as session:
            if pose_case:
                from data import action4_pose_formal_adapter as pose_adapter
                if render_scope is None:pose_proof=pose_adapter.pre_render(session,entry,parameters,rows[entry['sample_id']],plan,runtime)
                else:pose_proof=pose_adapter.pre_render(session,entry,parameters,rows[entry['sample_id']],plan,runtime,render_scope=render_scope)
            result=pixels.render_frame(scene=session.scene,entry=deepcopy(entry),targets=session.targets(),
                canonical=session.contract['canonical_definition'],K=parameters['K_native'],
                data_out=output_dir,runtime_out=runtime,output_role='dataset_candidate',
                source_identities=deepcopy(plan['asset_identities']),render_settings=(deepcopy(plan['recipe']['render']) if render_scope is None else render_scope.render_settings()),
                verify_scene=session.verifier,probe_only=False)
            expected_calls=1+(1 if entry['target_count'] else 0)+entry['target_count']
            if result['render_calls']!=expected_calls or result['render_attempts']!=expected_calls or result['rendered'] is not True or result['restoration_verified'] is not True or result['source_identities_verified'] is not True:
                raise ValueError('Single-frame pixel budget/restoration/source verification failed')
            record=result['record']
            if pose_case:record['pose_controls_proof']=pose_adapter.finish_record(parameters,record,pose_proof)
            if parameters.get('schema')=='action4.modular_geometric_execution.v1':
                from data import action4_modular_geometric_consumer as geometric
                geometry=session.contract['geometric_effect_geometry']
                collision=session.contract['actor_collision_diagnostics']
                check=geometric.verify_record(parameters,geometry,record,collision)
                record['geometric_effect_proof']={'geometry':geometry,'collision':collision,'check':check}
                if check.get('effect_condition_pass') is not True:raise ValueError('Actual geometric family effect failed')
            record['asset_backend']=BACKEND
            record['modular_backend']={'entry_sha256':_digest(entry_identity(entry)),
                'plan_sha256':plan['_runtime_plan_sha256'], 'recipe_sha256':plan['recipe_sha256'],
                'parameters_ref':rows[entry['sample_id']]['parameters_ref'],
                'parameters_sha256':_digest(parameters),
                'source_context_sha256':context['source_context_sha256'],
                'parameter_counts_as_formal_coverage':parameters['augmentation']['counts_as_formal_coverage'],
                'base_kind':'palmar_wearing',
                'scene_contract':deepcopy(session.contract),
                'pixel_receipt':str((runtime/'samples'/entry['sample_id']/'render_receipt.json').relative_to(ROOT))}
            if render_scope is None:validate_record(record,entry,plan['_runtime_plan_sha256'],plan,rows)
            else:render_scope.validate_record(record,entry)
            # Pixel receipt preserves its original child record; manifest is the thin
            # enriched adapter record, and both remain immutable candidates.
            synth._write(runtime/'samples'/entry['sample_id']/'backend_record.json',record)
        return record
    finally:
        if pose_case:
            if session is not None and (not session.restored or not session.base_session.restored):raise ValueError('Pose formal wearing/original restoration incomplete')
            sampling.verify_sources(context)
            if render_scope is None:
                if asset_identities(plan['recipe'])!=source_before:raise ValueError('Pose formal source closure changed during actual/render transaction')
            else:render_scope.after_scene()


# Explicit run-local validation; no global verification cache.
class _PlanIndex:
    def __init__(self):
        from collections import Counter
        self.events=Counter();self.identities={};self.documents={};self.raw_json={};self.release_data=None;self.plan=None
        self._authority=None;self._authority_indices={};self._authority_checked=set();self._pose_checked=set();self._bank_dependencies={};self._sampling_context=None;self._sealed=False
        import os,threading
        self._typed_token=object();self._typed_pid=os.getpid();self._typed_thread=threading.get_ident();self._typed_scopes={};self._validation_cache={};self._validation_source_raw=None
    def _guard(self):
        import os,threading
        if self._sealed:raise ValueError('Opening index is sealed')
        if os.getpid()!=self._typed_pid or threading.get_ident()!=self._typed_thread:raise ValueError('Typed index owner mismatch')
    @staticmethod
    def path(reference):
        if not isinstance(reference,dict) or set(reference)!={'path','sha256'} or not isinstance(reference['sha256'],str) or len(reference['sha256'])!=64:raise ValueError('Exact SHA reference required')
        name=Path(reference['path'])
        if name.is_absolute() or '..' in name.parts:raise ValueError('Root-relative reference only')
        return synth._path(name,exists=True)
    def _register(self,reference):
        old=self.identities.get(reference['path'])
        if old is not None and old!=reference['sha256']:raise ValueError('Conflicting immutable reference')
        self.identities[reference['path']]=reference['sha256']
    def check_ref(self,reference):
        self._guard();path=self.path(reference)
        if reference['path'] not in self.identities:
            raw=path.read_bytes();self.events['reference_byte_reads']+=1
            if hashlib.sha256(raw).hexdigest()!=reference['sha256']:raise ValueError('Opening source SHA mismatch: '+reference['path'])
        self._register(reference);return path
    def read_ref(self,reference):
        self._guard();path=self.path(reference);self._register(reference)
        if reference['path'] not in self.documents:
            raw=path.read_bytes();self.events['reference_byte_reads']+=1
            if hashlib.sha256(raw).hexdigest()!=reference['sha256']:raise ValueError('JSON bytes differ from reference SHA: '+reference['path'])
            self.raw_json[reference['path']]=raw;self.documents[reference['path']]=json.loads(raw);self.events['JSON_parses']+=1
        return self.documents[reference['path']]
    def hash_path(self,path):
        self._guard();p=synth._path(path,exists=True);name=str(p.relative_to(synth.ROOT))
        if name in self.identities:return self.identities[name]
        raw=p.read_bytes();self.events['reference_byte_reads']+=1;digest=hashlib.sha256(raw).hexdigest();self._register({'path':name,'sha256':digest});return digest
    def bind_plan(self,plan):self.plan=plan
    def require_plan(self,plan):
        self._guard()
        if plan is not self.plan and plan!=self.plan:raise ValueError('Opening index belongs to another plan')
    def require_recipe(self,recipe):
        self._guard()
        if self.plan is None or (recipe is not self.plan['recipe'] and recipe!=self.plan['recipe']):raise ValueError('Opening index belongs to another recipe')
    def plan_authority(self,plan):
        self.require_plan(plan)
        if self._authority is None:
            self.events['authority_builds']+=1;self.events['authority_entries_built']+=plan['count'];self._authority=synth.build_entry_authority(plan['recipe'],plan['count'],plan['seed'])
        return self._authority
    def authority_entry(self,reference,sample_id):
        self._guard();key=(reference['path'],reference['sha256'])
        if key not in self._authority_indices:
            authority=self.read_ref(reference);index={}
            for entry in authority['entries']:
                sid=entry['sample_id']
                if sid in index:raise ValueError('Duplicate authority sample ID')
                index[sid]=entry
            self._authority_indices[key]=index;self.events['authority_index_builds']+=1;self.events['authority_index_entry_visits']+=len(index)
        return self._authority_indices[key].get(sample_id)
    def check_plan_authority(self,reference,plan):
        self.require_plan(plan);key=(reference['path'],reference['sha256'])
        if plan.get('entry_authority_ref')!=reference:raise ValueError('Pose authority differs from formal plan')
        if key not in self._authority_checked:
            authority=self.read_ref(reference)
            if authority!=self.plan_authority(plan) or authority['entries']!=plan['entries']:raise ValueError('Pose authority differs from actual formal plan allocation')
            self._authority_checked.add(key)
    def _validation_source(self,context):
        self._guard()
        if context is None:return None
        source={'base_kind':context.get('base_kind'),'source_context':context['source_context'],'source_context_sha256':context['source_context_sha256']}
        if _digest(source['source_context'])!=source['source_context_sha256']:raise ValueError('Scoped source self identity mismatch')
        raw=json.dumps(source,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if self._validation_source_raw is None:self._validation_source_raw=raw
        elif self._validation_source_raw!=raw:raise ValueError('Validation scope belongs to another source context')
        return source
    def pose_context(self,context):
        if context is None:return None
        from data import action4_modular_pose_controls as controls
        self._validation_source(context)
        return controls._ScopedPoseContext(context,self,self._typed_token)
    def validation_once(self,kind,identity,context,producer):
        self._guard()
        if kind not in ('composition_descriptor','composition_dependencies','historical_rebind'):raise ValueError('Unknown typed validation memo kind')
        source=self._validation_source(context)
        key=(kind,_digest(identity),_digest(source))
        if key in self._validation_cache:
            raw,digest=self._validation_cache[key]
            if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Immutable validation result changed')
            self.events['validation_memo_hits_'+kind]+=1
            return json.loads(raw)
        result=producer()
        raw=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        self._validation_cache[key]=(raw,hashlib.sha256(raw).hexdigest())
        self.events['validation_memo_computations_'+kind]+=1
        return json.loads(raw)
    def _typed_scope(self,reference,context):
        from data import action4_modular_pose_controls as controls
        self._guard();self.check_ref(reference)
        key=(reference['path'],reference['sha256'],context['source_context_sha256'])
        if key not in self._typed_scopes:
            self._typed_scopes[key]=controls._TypedPoseScope(reference,context,self._typed_token)
            self.events['typed_static_loads']+=1
        scope=self._typed_scopes[key]
        scope._payload(reference,context,self._typed_token)
        return scope
    def validate_pose_once(self,reference,parameters,context):
        self._guard()
        # Cached SHA must never authorize a different caller-supplied object.
        self.read_ref(reference)
        if json.loads(self.raw_json[reference['path']])!=parameters:raise ValueError('Pose parameters differ from referenced immutable JSON')
        key=(reference['path'],reference['sha256'],context['source_context_sha256'])
        scope=self._typed_scope(parameters['provenance']['bank_config_ref'],context)
        if key not in self._pose_checked:
            scope.validate(parameters,context,self._typed_token)
            self._pose_checked.add(key);self.events['typed_pose_validations']+=1
    def bank_dependencies(self,reference,context):
        self._guard();scope=self._typed_scope(reference,context)
        key=(reference['path'],reference['sha256'],context['source_context_sha256'])
        if key not in self._bank_dependencies:
            self._bank_dependencies[key]=True;self.events['typed_bank_dependency_resolutions']+=1
        return scope.dependencies(reference,context,self._typed_token)
    def sampling_context(self,spec,source,release):
        from data import action4_modular_scene_sampling as sampling
        if self._sampling_context is None:
            context=sampling.prepare_base(self.check_ref(spec['scene_config_ref']));self.events['sampling_prepare_base']+=1
            if context.get('base_kind')!='palmar_wearing' or context['source_context']!=source or context['source_context_sha256']!=release['source_context_sha256']:raise ValueError('Prepared context differs from release')
            sampling.verify_sources(context);self._sampling_context=context
        return self._sampling_context
    def seal(self):
        from types import MappingProxyType
        self._guard()
        for raw,digest in self._validation_cache.values():
            if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Immutable validation result changed at seal')
        for scope in self._typed_scopes.values():scope._seal_owned(self._typed_token)
        for name,raw in self.raw_json.items():
            if json.loads(raw)!=self.documents[name]:raise ValueError('Validator mutated opening snapshot')
        self._sealed=True;self.identities=MappingProxyType(dict(self.identities))


def _current_reference(reference):
    path=_PlanIndex.path(reference);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=reference['sha256']:raise ValueError('Live reference SHA drift: '+reference['path'])
    return raw


class _FrozenRows:
    def __init__(self,rows):
        from types import MappingProxyType
        self.__rows=MappingProxyType({k:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode() for k,v in rows.items()})
    def __getitem__(self,key):return json.loads(self.__rows[key])
    def get(self,key,default=None):return self[key] if key in self.__rows else default
    def __len__(self):return len(self.__rows)
    def __iter__(self):return iter(self.__rows)
    def values(self):return (self[key] for key in self)
class ValidatedPlanContext:
    """Immutable run-local metadata. All record results remain provisional until close."""
    def __init__(self,plan_ref,run_token):
        import os,threading,uuid
        if run_token is None:raise ValueError('Explicit run token required')
        self.__token=run_token;self.__pid=os.getpid();self.__thread=threading.get_ident();self.__run_id=uuid.uuid4().hex
        self.__closed=False;self.__poisoned=False;self.__active=None;self.__errors=[];self.__extra_refs={};self.__receipt=None
        self.__plan_ref=deepcopy(plan_ref);index=_PlanIndex();plan=index.read_ref(plan_ref);index.bind_plan(plan)
        required={'schema_version','status','recipe','recipe_path','recipe_sha256','entries','count','seed','split_counts','asset_identities','entry_authority_ref','synthesis_code_sha256','render_base_sha256'}
        if not required<=set(plan) or plan['schema_version']!=synth.SCHEMA or plan['status']!='render_candidates_pending_acceptance' or plan['recipe'].get('asset_backend')!=BACKEND:raise ValueError('Complete frozen formal plan required for batch context')
        if index.read_ref({'path':plan['recipe_path'],'sha256':plan['recipe_sha256']})!=plan['recipe']:raise ValueError('Formal recipe bytes differ from plan')
        if plan['synthesis_code_sha256']!=index.hash_path(synth.__file__) or plan['render_base_sha256']!=index.hash_path('src/data/action4_render.py'):raise ValueError('Plan runtime source identity changed')
        synth.validate_recipe(plan['recipe'],_validation_index=index);validate_plan(plan,_index=index)
        context,rows=prepare_context(plan['recipe'],_index=index);assets=asset_identities(plan['recipe'],_index=index)
        if assets!=plan['asset_identities']:raise ValueError('Fresh source closure differs from frozen plan')
        from data import action4_pose_formal_adapter as pose_adapter
        for row in rows.values():
            if 'pose_execution_ref' in row:pose_adapter.validate_binding(row,context,plan,_index=index)
        index.seal();self.__index=index;self.__plan=deepcopy(plan);self.__plan_bytes=json.dumps(plan,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        self.__rows=_FrozenRows(rows);self.__entries=_FrozenRows({e['sample_id']:e for e in plan['entries']});self.__parameters=_FrozenRows({sid:index.documents[row['parameters_ref']['path']] for sid,row in rows.items()});self.__bindings=_FrozenRows({sid:index.documents[row['pose_execution_ref']['path']] for sid,row in rows.items() if 'pose_execution_ref' in row})
        self.__assets=dict(assets);self.__record_checks=0
    def __getstate__(self):raise TypeError('Validated context cannot cross process/run boundary')
    def _poison_internal(self,error):
        self.__poisoned=True;self.__errors.append(repr(error));self.__receipt=None
    def _owner_guard(self):
        import os,threading
        if os.getpid()!=self.__pid or threading.get_ident()!=self.__thread:
            self._poison_internal('Cross-owner context operation');raise ValueError('Context run token/owner mismatch')
    @property
    def plan(self):self._owner_guard();return json.loads(self.__plan_bytes)
    @property
    def plan_ref(self):self._owner_guard();return deepcopy(self.__plan_ref)
    @property
    def closed(self):self._owner_guard();return self.__closed
    @property
    def poisoned(self):self._owner_guard();return self.__poisoned
    @property
    def opening_counts(self):self._owner_guard();return dict(self.__index.events)
    @property
    def receipt(self):
        self._owner_guard()
        if self.__receipt is None or self.__poisoned:raise ValueError('No successful closing receipt')
        return deepcopy(self.__receipt)
    def poison(self,error):self._owner_guard();self._poison_internal(error)
    def _guard(self,token,*,active=False):
        self._owner_guard()
        if token is not self.__token:
            self._poison_internal('Cross-run context token');raise ValueError('Context run token/owner mismatch')
        if self.__closed or self.__poisoned:raise ValueError('Open healthy plan context required')
        if bool(self.__active is not None)!=active:self.poison('Reentrant context');raise ValueError('Invalid reentrant/context record scope')
    def bound_plan(self,reference,run_token):
        self._guard(run_token,active=True)
        if reference!=self.__plan_ref:self.poison('Different plan reference');raise ValueError('Record belongs to a different frozen plan')
        return self.__plan
    def _require_supplied_plan(self,plan):
        if plan is not None and plan is not self.__plan:
            frozen={k:v for k,v in plan.items() if k not in ('_runtime_plan_sha256','_runtime_plan_ref')}
            if json.dumps(frozen,sort_keys=True,separators=(',',':'),allow_nan=False).encode()!=self.__plan_bytes:raise ValueError('Caller supplied another plan to the same run token')
    def validate_record(self,record,entry,plan_sha256,plan,release_rows,run_token):
        self._guard(run_token)
        try:
            self._require_supplied_plan(plan)
            if plan_sha256 is not None and plan_sha256!=self.__plan_ref['sha256']:raise ValueError('Foreign plan SHA')
            sid=record['sample_id'];expected=self.__entries.get(sid)
            if expected is None or entry_identity(record)!=expected or (entry is not None and entry_identity(entry)!=expected):raise ValueError('Record/full entry differs from immutable plan index')
            row=self.__rows[sid]
            if release_rows is not None and release_rows[sid]!=row:raise ValueError('Caller release row differs from immutable index')
            meta=record['modular_backend']
            if meta['plan_sha256']!=self.__plan_ref['sha256'] or meta['parameters_ref']!=row['parameters_ref']:raise ValueError('Record plan/parameter source differs from indexed authority')
            parameters=json.loads(_current_reference(row['parameters_ref']))
            if parameters!=self.__parameters[sid] or parameters['entry']!=expected:raise ValueError('Current parameters differ from sealed immutable snapshot')
            binding=None
            if 'pose_execution_ref' in row:
                proof=record['pose_controls_proof']
                if proof['plan_ref']!=self.__plan_ref or proof['binding_ref']!=row['pose_execution_ref']:raise ValueError('Record pose proof belongs to another plan/binding')
                binding=json.loads(_current_reference(row['pose_execution_ref']))
                if binding!=self.__bindings[sid]:raise ValueError('Current pose binding differs from sealed snapshot')
            self.__active=(record,row,parameters,binding);self.__record_checks+=1
            result=validate_record(record,expected,self.__plan_ref['sha256'],self.__plan,self.__rows,validated_plan=self,run_token=run_token,_in_context=True)
            self._guard(run_token,active=True)
            return result
        except BaseException as exc:self.poison(exc);raise
        finally:self.__active=None
    def active_record_values(self,record,run_token):
        self._guard(run_token,active=True)
        if record is not self.__active[0]:self.poison('Different active record');raise ValueError('Active record identity mismatch')
        return self.__plan,self.__rows,self.__active[2]
    def record_binding(self,row,record,run_token,plan):
        self._guard(run_token,active=True);self._require_supplied_plan(plan)
        if record is not self.__active[0] or row!=self.__active[1] or self.__active[3] is None:raise ValueError('Exact active pose binding required')
        return self.__active[2],self.__active[3]
    def worker_metadata(self,run_token):
        self._guard(run_token)
        return self.__index._sampling_context,self.__rows
    def worker_entry_inputs(self,entry,run_token):
        self._guard(run_token)
        try:
            e=entry_identity(entry);sid=e['sample_id'];expected=self.__entries.get(sid)
            if e!=expected:raise ValueError('Worker entry differs from complete indexed authority')
            row=self.__rows[sid];parameters=json.loads(_current_reference(row['parameters_ref']))
            if parameters!=self.__parameters[sid] or parameters['entry']!=expected:raise ValueError('Worker parameters differ from opening snapshot')
            binding=None
            if 'pose_execution_ref' in row:
                binding=json.loads(_current_reference(row['pose_execution_ref']))
                if binding!=self.__bindings[sid]:raise ValueError('Worker pose binding differs from opening snapshot')
            return row,parameters,binding
        except BaseException as exc:self.poison(exc);raise
    def parameters_for_acceptance(self,entry,run_token):
        self._guard(run_token)
        try:
            e=entry_identity(entry);sid=e['sample_id'];row=self.__rows[sid]
            if e!=self.__entries[sid]:raise ValueError('Acceptance entry differs from immutable plan')
            fresh=json.loads(_current_reference(row['parameters_ref']))
            if fresh!=self.__parameters[sid]:raise ValueError('Acceptance parameters changed')
            if fresh.get('schema')=='action4.modular_pose_controls.v1':return fresh
            return parameters_for(e,self.__index._sampling_context,self.__rows)
        except BaseException as exc:self.poison(exc);raise
    def generation_bindings(self,run_token):
        self._guard(run_token)
        plan=self.__plan;plan_ref=self.__plan_ref;context=self.__index._sampling_context
        self.track_reference(plan_ref,run_token)
        self.track_reference(CURRENT_PRESENCE_QUOTA_POLICY_REF,run_token)
        spec,source,_,_=self.__index.release_data
        geometry={path:{'path':path,'sha256':digest} for path,digest in source['frozen_refs'].items()
                  if Path(path).suffix.lower() in ('.npz','.blend','.stl')}
        canonical=context['wearing_context']['canonical_definition']['source_manifest']
        self.track_reference(canonical,run_token)
        if source['frozen_refs'].get(canonical['path'])!=canonical['sha256']:
            raise ValueError('Canonical original8 source must be in the frozen closure')
        geometry[canonical['path']]=canonical
        if not any(Path(path).suffix=='.blend' for path in geometry) or not any(Path(path).suffix=='.npz' for path in geometry):
            raise ValueError('Complete actual wearing geometry cache/asset references required')
        return {'schema':'action4.modular_generation_binding.v1','base_kind':'palmar_wearing',
                'plan':plan_ref,'recipe_sha256':plan['recipe_sha256'],
                'presence_quota_policy_ref':deepcopy(CURRENT_PRESENCE_QUOTA_POLICY_REF),
                'source_context_sha256':context['source_context_sha256'],
                'asset_identities_sha256':_digest(plan['asset_identities']),
                'sampling_bounds_ref':spec['sampling_bounds_ref'],
                'parameters_release_ref':spec['parameters_release_ref'],
                'geometry_refs':geometry,'planned_entries_sha256':_digest(plan['entries'])}
    def track_reference(self,reference,run_token):
        self._guard(run_token)
        try:
            expected=self.__index.identities.get(reference['path'],self.__extra_refs.get(reference['path']))
            if expected is not None and expected!=reference['sha256']:raise ValueError('Conflicting tracked evidence reference')
            _current_reference(reference);self.__extra_refs[reference['path']]=reference['sha256']
        except BaseException as exc:self.poison(exc);raise
    def close(self,run_token):
        import os,threading
        self._owner_guard()
        if self.__closed:self._poison_internal('Repeated context close');raise ValueError('Plan context already closed')
        if run_token is not self.__token or os.getpid()!=self.__pid or threading.get_ident()!=self.__thread or self.__active is not None:self.poison('Invalid close owner/state');raise ValueError('Idle same-run close required')
        self.__closed=True;failures=[];vector=dict(self.__index.identities);vector.update(self.__extra_refs)
        for path,digest in vector.items():
            try:_current_reference({'path':path,'sha256':digest})
            except BaseException as exc:failures.append(str(exc))
        closing=_PlanIndex()
        try:
            fresh=closing.read_ref(self.__plan_ref);closing.bind_plan(fresh)
            if fresh!=self.__plan:raise ValueError('Formal plan changed during batch')
            if asset_identities(fresh['recipe'],_index=closing)!=self.__assets:raise ValueError('Fresh closing dependency map differs from opening map')
            closing.seal()
        except BaseException as exc:failures.append(str(exc))
        if failures:self.poison(failures)
        if self.__poisoned:raise ValueError('Poisoned batch cannot close successfully; '+repr(self.__errors))
        self.__receipt={'status':'PASS_RUN_LOCAL_PLAN_CLOSING_INTEGRITY_ONLY','run_id':self.__run_id,'plan_ref':self.plan_ref,'record_checks':self.__record_checks,'opening_counts':self.opening_counts,'closing_counts':dict(closing.events),'closing_full_reference_count':len(vector),'dependency_map_unchanged':True,'source_vector_sha256':_digest(vector),'training_accepted':False}
        return self.receipt


class _ReportPublication:
    """Exclusive multi-report publication with verified, ownership-bounded rollback.

    A journal is evidence, never an acceptance receipt. No existing target is
    overwritten or removed; a concurrently changed new target is retained.
    """
    def __init__(self,pending,run_id):
        self.pending=list(pending);self.run_id=run_id;self.rows=[];self.created=[];self.temps=[];self.journal_path=None;self.journal_paths=[];self.journal=None;self.failure=None
    @staticmethod
    def _identity(st):return {'device':int(st.st_dev),'inode':int(st.st_ino)}
    @staticmethod
    def _read_regular(path):
        import os,stat
        path=Path(path)
        if synth._path(path)!=path:raise ValueError('Publication path resolution drift')
        fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
        try:
            before=os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):raise ValueError('Report target must be a regular file')
            parts=[]
            while True:
                part=os.read(fd,1024*1024)
                if not part:break
                parts.append(part)
            after=os.fstat(fd)
            if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('Report changed while reading')
            raw=b''.join(parts);return raw,{**_ReportPublication._identity(after),'sha256':hashlib.sha256(raw).hexdigest()}
        finally:os.close(fd)
    @staticmethod
    def _matches(path,proof):
        raw,current=_ReportPublication._read_regular(path)
        return all(current[k]==proof[k] for k in ('device','inode','sha256'))
    def preflight(self):
        seen={}
        for path,canonical in self.pending:
            path=Path(path)
            if synth._path(path)!=path:raise ValueError('Deferred report path drift')
            if path in seen:
                if seen[path]!=canonical:raise ValueError('Conflicting deferred reports target the same path')
                continue
            seen[path]=canonical
            parent=path.parent
            while not parent.exists() and not parent.is_symlink():parent=parent.parent
            if not parent.is_dir() or parent.is_symlink():raise ValueError('Report parent is not a verified directory')
            value=json.loads(canonical);payload=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
            row={'path':path,'canonical':canonical,'payload':payload,'expected_sha256':hashlib.sha256(payload).hexdigest()}
            if path.exists() or path.is_symlink():
                raw,proof=self._read_regular(path)
                if json.dumps(json.loads(raw),sort_keys=True,separators=(',',':'),allow_nan=False).encode()!=canonical:raise ValueError('Existing validation report no longer matches verified bytes')
                row.update(kind='existing',proof=proof)
            else:row['kind']='new'
            self.rows.append(row)
        paths=set(seen)
        if any(any(parent in paths for parent in path.parents) for path in paths):raise ValueError('One report target is another report parent')
    def _stage_one(self,row):
        import os,tempfile
        path=row['path'];path.parent.mkdir(parents=True,exist_ok=True)
        if synth._path(path)!=path:raise ValueError('Report parent changed before staging')
        fd,name=tempfile.mkstemp(prefix='.host-validation-'+self.run_id+'-',suffix='.pending',dir=path.parent)
        temp={'path':Path(name),'identity':self._identity(os.fstat(fd)),'expected':row['payload'],'complete':False};self.temps.append(temp);row['temp']=temp
        try:
            with os.fdopen(fd,'wb',closefd=True) as stream:
                n=stream.write(row['payload'])
                if n!=len(row['payload']):raise OSError('Incomplete staged report write')
                stream.flush();os.fsync(stream.fileno())
            raw,proof=self._read_regular(temp['path'])
            if proof['sha256']!=row['expected_sha256'] or any(proof[k]!=temp['identity'][k] for k in ('device','inode')):raise ValueError('Staged report bytes/identity changed')
            temp['complete']=True;temp['proof']=proof;row['proof']=proof
        except BaseException:
            # The temp identity was captured before any write; abort will only
            # remove its still-matching full bytes or a proved write-prefix.
            raise
    def _open_journal(self):
        import os,uuid
        if self.journal is not None:return
        if not self.pending:return
        folder=Path(self.pending[0][0]).parent/'.host_validation_transactions'
        if synth._path(folder)!=folder:raise ValueError('Failure journal path drift')
        folder.mkdir(parents=True,exist_ok=True)
        path=folder/(self.run_id+'_'+uuid.uuid4().hex+'.jsonl')
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
        self.journal_path=path;self.journal_paths.append(str(path.relative_to(synth.ROOT)));self.journal=os.fdopen(fd,'wb',closefd=True)
    def _event(self,event):
        import os
        self._open_journal()
        if self.journal is None:return
        payload=(json.dumps(event,sort_keys=True,allow_nan=False)+'\n').encode();n=self.journal.write(payload)
        if n!=len(payload):raise OSError('Incomplete publication journal write')
        self.journal.flush();os.fsync(self.journal.fileno())
    def _publish_one(self,row):
        import os
        if row['kind']!='new':return
        if synth._path(row['path'])!=row['path'] or not self._matches(row['temp']['path'],row['proof']):raise ValueError('Report/staged path drift before publication')
        # Hard-link publication is atomic and refuses an existing target.
        os.link(row['temp']['path'],row['path'],follow_symlinks=False)
        self.created.append({'path':row['path'],'proof':dict(row['proof'])})
        self._event({'event':'REPORT_LINKED_UNRECEIPTED','path':str(row['path'].relative_to(synth.ROOT)),'proof':row['proof'],'training_accepted':False})
    @staticmethod
    def _safe_remove(path,proof):
        import os
        try:
            if not _ReportPublication._matches(path,proof):return {'path':str(path),'removed':False,'reason':'concurrent_identity_or_content_change'}
            now=os.lstat(path)
            if _ReportPublication._identity(now)!={k:proof[k] for k in ('device','inode')}:return {'path':str(path),'removed':False,'reason':'identity_changed_before_unlink'}
            path.unlink()
            if path.exists() or path.is_symlink():return {'path':str(path),'removed':False,'reason':'parallel_replacement_after_owned_unlink'}
            return {'path':str(path),'removed':True,'reason':'owned_identity_and_SHA_verified'}
        except FileNotFoundError:return {'path':str(path),'removed':True,'reason':'already_absent'}
        except BaseException as exc:return {'path':str(path),'removed':False,'reason':'cleanup_error','error':repr(exc)}
    def _cleanup_temps(self):
        results=[]
        for temp in self.temps:
            try:
                raw,proof=self._read_regular(temp['path'])
                if any(proof[k]!=temp['identity'][k] for k in ('device','inode')) or not temp['expected'].startswith(raw):
                    results.append({'path':str(temp['path']),'removed':False,'reason':'temporary_file_changed_or_unproved_prefix'});continue
                if temp['complete'] and raw!=temp['expected']:
                    results.append({'path':str(temp['path']),'removed':False,'reason':'completed_temporary_content_changed'});continue
                results.append(self._safe_remove(temp['path'],proof))
            except FileNotFoundError:results.append({'path':str(temp['path']),'removed':True,'reason':'already_absent'})
            except BaseException as exc:results.append({'path':str(temp['path']),'removed':False,'reason':'temporary_cleanup_error','error':repr(exc)})
        return results
    def publish(self):
        self.preflight()
        for row in self.rows:
            if row['kind']=='new':self._stage_one(row)
        self._event({'event':'PREPARED_UNACCEPTED','run_id':self.run_id,'targets':[{'path':str(r['path'].relative_to(synth.ROOT)),'kind':r['kind'],'proof':r['proof']} for r in self.rows],'training_accepted':False})
        for row in self.rows:self._publish_one(row)
        for row in self.rows:
            if not self._matches(row['path'],row['proof']):raise ValueError('Published or preexisting report changed before completion')
        temporary=self._cleanup_temps()
        if not all(r['removed'] for r in temporary):raise RuntimeError('Temporary publication cleanup incomplete')
        self._event({'event':'ALL_REPORTS_WRITTEN_NOT_AN_ACCEPTANCE_RECEIPT','run_id':self.run_id,'report_count':len(self.rows),'training_accepted':False})
        if self.journal is not None:self.journal.close();self.journal=None
        # Final read follows all writing/cleanup, including checks on preexisting files.
        for row in self.rows:
            if not self._matches(row['path'],row['proof']):raise ValueError('Report changed during publication finalization')
        return {'status':'ALL_DEFERRED_REPORTS_PUBLISHED','new_report_count':len(self.created),'existing_reports_preserved':sum(r['kind']=='existing' for r in self.rows),'report_count':len(self.rows),'journal_path':str(self.journal_path.relative_to(synth.ROOT)) if self.journal_path else None,'reports':[{'path':str(r['path'].relative_to(synth.ROOT)),'proof':r['proof']} for r in self.rows]}
    def abort(self,error):
        if self.failure is not None:return deepcopy(self.failure)
        rollback=[self._safe_remove(r['path'],r['proof']) for r in reversed(self.created)];temps=self._cleanup_temps()
        failure={'status':'FAILED_PUBLICATION_NO_PASS_RECEIPT','run_id':self.run_id,'error':repr(error),'created_reports':len(self.created),'rollback':rollback,'temporary_cleanup':temps,'rollback_complete':all(r['removed'] for r in [*rollback,*temps]),'existing_reports_modified':False,'journal_saved':False,'training_accepted':False}
        try:
            self._open_journal()
            failure['journal_saved']=self.journal is not None;failure['journal_paths']=list(self.journal_paths)
            self._event({'event':'FAILED','failure':failure})
            if self.journal is not None:self.journal.close();self.journal=None
            failure['journal_saved']=self.journal_path is not None;failure['journal_path']=str(self.journal_path.relative_to(synth.ROOT)) if self.journal_path else None
        except BaseException as journal_error:
            failure['journal_saved']=False;failure['journal_paths']=list(self.journal_paths);failure['journal_error']=repr(journal_error);failure['journal_path']=str(self.journal_path.relative_to(synth.ROOT)) if self.journal_path else None
            if self.journal is not None:
                try:self.journal.close()
                except BaseException:pass
                self.journal=None
        self.failure=deepcopy(failure);return failure


class HostValidationSession:
    """Owner-checked host lifecycle; PASS exists only after complete publication."""
    def __init__(self,plan_ref):
        import os,threading
        self.__pid=os.getpid();self.__thread=threading.get_ident();self.__token=object();self.__context=ValidatedPlanContext(plan_ref,self.__token)
        self.__entered=False;self.__closed=False;self.__busy=False;self.__pending=[];self.__state='NEW';self.__receipt=None;self.__failure=None
    def _invalidate(self,error):
        self.__receipt=None;self.__context._poison_internal(error)
    def _owner_guard(self):
        import os,threading
        if os.getpid()!=self.__pid or threading.get_ident()!=self.__thread:
            if self.__state not in ('SUCCEEDED','FAILED'):self._invalidate('Cross-owner public host operation')
            raise ValueError('Host session owner/run mismatch')
    def _guard(self):
        self._owner_guard()
        if self.__state!='OPEN' or self.__busy or self.__context.poisoned:self._invalidate('Invalid host-session state');raise ValueError('Open non-reentrant healthy host session required')
    @property
    def plan(self):self._guard();return self.__context.plan
    @property
    def plan_ref(self):self._guard();return self.__context.plan_ref
    @property
    def closed(self):self._owner_guard();return self.__closed
    @property
    def receipt(self):
        self._owner_guard()
        if self.__state!='SUCCEEDED' or self.__receipt is None or self.__context.poisoned:raise ValueError('No successful published host receipt')
        return deepcopy(self.__receipt)
    @property
    def failure_info(self):self._owner_guard();return deepcopy(self.__failure)
    @property
    def opening_counts(self):self._guard();return self.__context.opening_counts
    def poison(self,error):
        self._owner_guard();self._invalidate(error)
        if self.__state=='SUCCEEDED':self.__state='FAILED'
    def require_plan(self,reference):
        self._guard()
        if reference!=self.__context.plan_ref:self._invalidate('Different shard plan');raise ValueError('Host session cannot accept another plan')
    def validate_record(self,record,entry=None):
        self._guard();self.__busy=True
        try:return validate_record(record,entry,validated_plan=self.__context,run_token=self.__token)
        except BaseException as exc:self._invalidate(exc);raise
        finally:self.__busy=False
    def validate_host_record(self,record):
        from data import action4_modular_pixels as pixels
        self.validate_record(record);self._guard();self.__busy=True
        try:
            result=pixels.validate_host_pngs(record)
            for item in result['files']:self.__context.track_reference({'path':item['path'],'sha256':item['sha256']},self.__token)
            if self.__context.poisoned:raise ValueError('Host validation poisoned during PNG checks')
            return result
        except BaseException as exc:self._invalidate(exc);raise
        finally:self.__busy=False
    def read_tracked_json(self,path):
        self._guard()
        try:
            path=synth._path(path,exists=True);raw=path.read_bytes();self.track_reference({'path':str(path.relative_to(synth.ROOT)),'sha256':hashlib.sha256(raw).hexdigest()});return json.loads(raw)
        except BaseException as exc:self._invalidate(exc);raise
    def parameters_for_acceptance(self,entry):
        self._guard()
        try:return self.__context.parameters_for_acceptance(entry,self.__token)
        except BaseException as exc:self._invalidate(exc);raise
    def generation_bindings(self):
        self._guard()
        try:return self.__context.generation_bindings(self.__token)
        except BaseException as exc:self._invalidate(exc);raise
    def track_reference(self,reference):
        self._guard();self.__context.track_reference(reference,self.__token)
    def defer_report(self,path,result):
        self._guard()
        try:self.__pending.append((synth._path(path),json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False).encode()))
        except BaseException as exc:self._invalidate(exc);raise
    def close(self):
        self._owner_guard()
        if self.__state!='OPEN' or self.__busy:self._invalidate('Invalid close state');raise ValueError('Idle open host session required for close')
        self.__state='CLOSING';self.__closed=True;publisher=None
        try:
            integrity=self.__context.close(self.__token)
            publisher=_ReportPublication(self.__pending,integrity['run_id']);publication=publisher.publish()
            if self.__context.poisoned:raise ValueError('Session poisoned during report publication')
            receipt={**integrity,'status':'PASS_HOST_VALIDATION_SESSION_CLOSED_AND_PUBLISHED','publication':publication}
            self.__pending.clear();self.__receipt=receipt;self.__state='SUCCEEDED'
            return self.receipt
        except BaseException as exc:
            self._invalidate(exc);self.__state='FAILED'
            self.__failure=publisher.abort(exc) if publisher is not None else {'status':'FAILED_INTEGRITY_NO_PUBLICATION','error':repr(exc),'rollback_complete':True,'created_reports':0,'journal_saved':False}
            try:exc.failure_info=deepcopy(self.__failure)
            except BaseException:pass
            raise
    def __enter__(self):
        self._owner_guard()
        if self.__state!='NEW':self._invalidate('Session reentry');raise ValueError('One host-session entry only')
        if self.__context.poisoned:raise ValueError('Poisoned session cannot enter')
        self.__entered=True;self.__state='OPEN';return self
    def __exit__(self,kind,error,tb):
        self._owner_guard()
        if error is not None:self.poison(error)
        if not self.__closed:
            try:self.close()
            except BaseException as final:raise RuntimeError('Host closing failed; primary='+repr(error)+'; closing='+repr(final)) from (error or final)
        return False

def _validate_generation_release_batched(plan,plan_ref,records,acceptance,validation_session):
    """Reachable promotion: candidates stay false; an independent reviewed
    generation release AND existing native/128/six-criteria checks authorize the
    assembled manifest. Every original record/parameter is checked before that.
    """
    from data import action4_modular_scene_sampling as sampling
    validation_session.require_plan(plan_ref)
    if validation_session.plan!=plan:raise ValueError('Acceptance plan differs from validated session')
    expected=validation_session.generation_bindings()
    release_ref=acceptance.get('generation_release_ref')
    if release_ref is None:raise ValueError('Reviewed generation release required; candidate flags cannot promote data')
    validation_session.track_reference(release_ref)
    release=synth._read(_ref(release_ref))
    if set(release)!={'schema','status','bindings','reviews','presence_quota_policy_ref'} or release['schema']!='action4.modular_generation_release.v1' or release['status']!='REVIEWED_FOR_BOUND_PLAN' or release['bindings']!=expected:
        raise ValueError('Generation release differs from fixed plan/source/geometry/sampling bounds')
    validate_presence_quota(plan,release['presence_quota_policy_ref'])
    if release['presence_quota_policy_ref']!=expected['presence_quota_policy_ref']:raise ValueError('Generation review quota policy binding differs')
    if set(release['reviews'])!={'spec','root'} or release['reviews']['spec']['path']==release['reviews']['root']['path']:
        raise ValueError('Distinct explicit SPEC and root generation reviews required')
    for role,ref in release['reviews'].items():
        validation_session.track_reference(ref)
        review=synth._read(_ref(ref))
        if review.get('schema')!='action4.modular_generation_review.v1' or review.get('role')!=role or review.get('decision')!='approved_generation' or not review.get('reviewer') or not review.get('observations') or review.get('binding_sha256')!=_digest(expected):
            raise ValueError('Generation '+role+' review does not approve these exact bindings')
    by_id={r['sample_id']:r for r in records}
    if len(by_id)!=len(records) or set(by_id)!={e['sample_id'] for e in plan['entries']}:
        raise ValueError('Generation acceptance needs every record exactly once')
    # The actual consumer rechecks every family and bounded control. Six pending
    # effects cannot be waived by a review document or relabeled as wall_rear.
    for entry in plan['entries']:
        validation_session.parameters_for_acceptance(entry)
        validation_session.validate_record(by_id[entry['sample_id']],entry)
        validate_formal_targets(by_id[entry['sample_id']],entry,plan['recipe']['quality_gate'])
    return release_ref



def validate_generation_release_batched(plan,plan_ref,records,acceptance,validation_session):
    try:return _validate_generation_release_batched(plan,plan_ref,records,acceptance,validation_session)
    except BaseException as exc:validation_session.poison(exc);raise


class _RenderEntryScope:
    # Constructed only by RenderBatch; every method checks its active owner.
    def __init__(self,batch,entry,row,parameters,binding):
        if type(batch) is not RenderBatch:raise ValueError('Concrete live RenderBatch owner required')
        self.__batch=batch;self.__entry=deepcopy(entry);self.__row=deepcopy(row);self.__parameters=deepcopy(parameters);self.__binding=deepcopy(binding)
    def _guard(self):return self.__batch._check_scope(self)
    def parameters_for(self,entry,plan,context,rows):
        self._guard();self.__batch._check_inputs(entry,plan,context,rows,self.__entry)
        if self.__parameters.get('schema')=='action4.modular_pose_controls.v1':
            # Full typed validation occurred at open and is still performed by
            # unchanged sampling.create_scene_for_entry for every real scene.
            return deepcopy(self.__parameters)
        result=parameters_for(entry,context,rows)
        if result!=self.__parameters:raise ValueError('Worker typed parameters differ from frozen bytes')
        return result
    def pre_render_binding(self,session,entry,parameters,row,plan):
        self._guard()
        if entry!=self.__entry or row!=self.__row or parameters!=self.__parameters or self.__binding is None:raise ValueError('Exact worker pose pre-render scope required')
        self.__batch._check_pre_render(session,plan)
        self.after_scene()  # Fresh per-row bytes before the actual geometry gate.
        return self.__batch._plan_ref_for_scope(self)
    def render_settings(self):
        self._guard();return self.__batch._settings_for_scope(self)
    def validate_record(self,record,entry):
        self._guard()
        if entry!=self.__entry:raise ValueError('Worker record belongs to another entry')
        return self.__batch._record_for_scope(self,record,entry)
    def after_scene(self):
        self._guard();row,p,b=self.__batch._fresh_for_scope(self,self.__entry)
        if row!=self.__row or p!=self.__parameters or b!=self.__binding:raise ValueError('Worker entry inputs changed during scene transaction')


class RenderBatch:
    """Explicit owner-local worker metadata reuse; no scene or pixel cache.

    Opening/closing traverse complete plan/release/source metadata. Each entry
    rechecks its parameter/binding bytes and keeps existing scene, typed, actual
    geometry, pixel-source/mask and restore paths. Those leaves may still scan
    authority/full source maps; this is not whole-worker linear-time evidence.
    """
    def __init__(self,plan_ref,entries,*,expected_plan=None,gpu_uuid=None):
        import os,threading
        self.__pid=os.getpid();self.__thread=threading.get_ident();self.__token=object()
        self.__context=ValidatedPlanContext(plan_ref,self.__token)
        self.__plan=self.__context.plan;self.__ref=self.__context.plan_ref
        if expected_plan is not None and {k:v for k,v in expected_plan.items() if k not in ('_runtime_plan_ref','_runtime_plan_sha256')}!=self.__plan:
            self.__context.poison('Foreign expected worker plan');raise ValueError('Worker batch belongs to another plan')
        self.__plan['_runtime_plan_ref']=deepcopy(self.__ref);self.__plan['_runtime_plan_sha256']=self.__ref['sha256']
        self.__sampling,self.__rows=self.__context.worker_metadata(self.__token)
        self.__entries={};self.__done=set();self.__attempted=set();self.__scope=None
        self.__state='NEW';self.__receipt=None
        for entry in entries:
            e=entry_identity(entry);sid=e['sample_id']
            if sid in self.__entries:raise ValueError('Duplicate worker batch entry')
            row,p,b=self.__context.worker_entry_inputs(e,self.__token);self.__entries[sid]=deepcopy(e)
        self.__settings=deepcopy(self.__plan['recipe']['render'])
        if self.__settings['device']=='CPU':
            if gpu_uuid is not None:raise ValueError('CPU worker cannot accept GPU')
        else:
            if gpu_uuid not in self.__settings.get('gpu_uuids',[]):raise ValueError('Worker GPU differs from frozen allowlist')
            self.__settings['worker_gpu_uuid']=gpu_uuid
    def __getstate__(self):raise TypeError('RenderBatch cannot cross process/run')
    def _owner(self):
        import os,threading
        if os.getpid()!=self.__pid or threading.get_ident()!=self.__thread:
            self.__receipt=None;self.__context._poison_internal('Cross-owner render batch')
            raise ValueError('Worker batch PID/thread owner mismatch')
    def _healthy(self):
        self._owner()
        if self.__state!='OPEN' or self.__context.poisoned:
            self.poison('Invalid worker batch state');raise ValueError('Open healthy worker batch required')
    def poison(self,error):
        self._owner();self.__receipt=None;self.__context.poison(error)
    def _check_scope(self,scope):
        self._healthy()
        if self.__scope is not scope:
            self.poison('Foreign or expired worker entry scope');raise ValueError('Active worker entry scope required')
    def _check_inputs(self,entry,plan,context,rows,expected):
        if entry!=expected or plan is not self.__plan or context is not self.__sampling or rows is not self.__rows:
            self.poison('Different worker scope input');raise ValueError('Frozen worker plan/context/entry identity')
    def _check_pre_render(self,session,plan):
        if plan is not self.__plan or session.context is not self.__sampling:
            self.poison('Different worker pre-render plan/context');raise ValueError('Actual scene not owned by worker scope')
    def _plan_ref_for_scope(self,scope):self._check_scope(scope);return deepcopy(self.__ref)
    def _settings_for_scope(self,scope):self._check_scope(scope);return deepcopy(self.__settings)
    def _fresh_for_scope(self,scope,entry):self._check_scope(scope);return self.__context.worker_entry_inputs(entry,self.__token)
    def _record_for_scope(self,scope,record,entry):
        self._check_scope(scope)
        return self.__context.validate_record(record,entry,self.__ref['sha256'],None,None,self.__token)
    def render_entry(self,entry,output_dir):
        self._healthy()
        if self.__scope is not None:self.poison('Reentrant worker entry');raise ValueError('Non-reentrant worker entry required')
        try:
            e=entry_identity(entry);sid=e['sample_id']
            if self.__entries.get(sid)!=e or sid in self.__attempted:raise ValueError('Unselected or repeated worker entry')
            self.__attempted.add(sid);row,p,b=self.__context.worker_entry_inputs(e,self.__token)
            scope=_RenderEntryScope(self,e,row,p,b);self.__scope=scope
            record=_render_prepared_entry(self.__plan,e,output_dir,self.__sampling,self.__rows,render_scope=scope)
            scope.after_scene();self.__done.add(sid);return record
        except BaseException as exc:self.poison(exc);raise
        finally:self.__scope=None
    @property
    def receipt(self):
        self._owner()
        if self.__state!='CLOSED' or self.__receipt is None or self.__context.poisoned:raise ValueError('No successful worker batch closing receipt')
        return deepcopy(self.__receipt)
    def close(self):
        self._owner()
        if self.__state!='OPEN' or self.__scope is not None:
            self.poison('Repeated/active worker close');raise ValueError('Idle open worker batch close required')
        self.__state='CLOSING'
        if self.__done!=set(self.__entries):self.poison('Incomplete worker batch')
        try:
            from data import action4_modular_scene_sampling as sampling
            try:sampling.verify_sources(self.__sampling)
            except BaseException as exc:self.poison(exc)
            integrity=self.__context.close(self.__token)
            self.__receipt={**integrity,'status':'PASS_WORKER_METADATA_CLOSED_MANIFEST_NOT_YET_PUBLISHED','completed_sample_ids':sorted(self.__done),'training_accepted':False}
            self.__state='CLOSED';return self.receipt
        except BaseException as exc:self.__state='FAILED';self.__receipt=None;raise
    def __enter__(self):
        self._owner()
        if self.__state!='NEW' or self.__context.poisoned:self.poison('Worker batch reentry');raise ValueError('New healthy worker batch required')
        self.__state='OPEN';return self
    def __exit__(self,kind,error,tb):
        if error is not None:self.poison(error)
        if self.__state=='OPEN':
            try:self.close()
            except BaseException as final:raise RuntimeError('Worker closing failed; primary='+repr(error)+'; closing='+repr(final)) from (error or final)
        return False


def _render_composition_entry(plan,entry,output_dir,context,rows,parameters,*,render_scope=None):
    """Formal v2 transaction: original gate -> layers -> pixels -> layers -> base.
    No enriched candidate record is published before every cleanup/source check.
    """
    from data import action4_modular_scene_sampling as sampling
    from data import action4_pose_formal_adapter as adapter
    from data import action4_formal_composition as composer
    runtime=synth._run_output_dir(output_dir);session=None;record=None;primary=None;cleanup=[]
    row=rows[entry['sample_id']];source_before=deepcopy(plan['asset_identities'])
    try:
        with sampling.create_scene_for_entry(context,parameters) as session:
            proof=adapter.pre_render(session,entry,parameters,row,plan,runtime,render_scope=render_scope)
            settings=deepcopy(plan['recipe']['render']) if render_scope is None else render_scope.render_settings()
            result,proof=composer.render_composition(session,entry,parameters,row,plan,runtime,proof,settings,output_dir)
            record=result['record'];record['pose_controls_proof']=adapter.finish_record(parameters,record,proof)
            record['asset_backend']=BACKEND
            record['modular_backend']={'entry_sha256':_digest(entry_identity(entry)),'plan_sha256':plan['_runtime_plan_sha256'],'recipe_sha256':plan['recipe_sha256'],'parameters_ref':row['parameters_ref'],'parameters_sha256':_digest(parameters),'source_context_sha256':context['source_context_sha256'],'parameter_counts_as_formal_coverage':parameters['augmentation']['counts_as_formal_coverage'],'base_kind':'palmar_wearing','scene_contract':deepcopy(session.contract),'pixel_receipt':str((runtime/'samples'/entry['sample_id']/'render_receipt.json').relative_to(ROOT))}
    except BaseException as exc:primary=exc
    try:
        if session is not None and (not session.restored or not session.base_session.restored):raise ValueError('Composition base/original source restoration incomplete')
        sampling.verify_sources(context)
        if render_scope is None:
            if asset_identities(plan['recipe'])!=source_before:raise ValueError('Composition formal source closure changed')
        else:render_scope.after_scene()
    except BaseException as exc:cleanup.append(repr(exc))
    if primary is not None or cleanup:
        synth._write(runtime/'composition_failure.json',{'status':'FAILED_RETAINED_NO_RETRY','primary':None if primary is None else repr(primary),'cleanup':cleanup,'training_accepted':False,'dataset_generation_allowed':False})
        raise RuntimeError('Formal composition transaction failed: '+repr(primary)+'; cleanup='+repr(cleanup)) from primary
    if render_scope is None:validate_record(record,entry,plan['_runtime_plan_sha256'],plan,rows)
    else:render_scope.validate_record(record,entry)
    synth._write(runtime/'samples'/entry['sample_id']/'backend_record.json',record)
    return record
