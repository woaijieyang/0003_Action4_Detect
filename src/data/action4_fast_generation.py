"""Fast execution orchestration over frozen Action4 physics and parameter plan.
No model reconstruction, label substitution or unreviewed legacy fallback.
Runtime memo adapters are explicit and byte-equivalent on unchanged inputs.
"""
from pathlib import Path
from copy import deepcopy
import hashlib, json, shutil, time
from . import action4_synthesis as synth, action4_modular_synthesis_backend as b
from . import action4_pose_formal_adapter as adapter, action4_modular_pose_controls as controls, action4_modular_pixels as pixels, action4_modular_render as camera
from .action4_execution_cache import FastRenderBatch, FastHostSession
from .action4_persistent_base import PersistentWearingBase
from .action4_runtime_memo import RuntimeMemo
ROOT = synth.ROOT
BACKEND = b.BACKEND
_digest = b._digest
entry_identity = b.entry_identity
parameters_for = b.parameters_for
asset_identities = b.asset_identities
validate_record = b.validate_record
_read = synth._read
_path = synth._path
_hash = synth._hash
_write = synth._write
_run_output_dir = synth._run_output_dir
select_shard_entries = synth.select_shard_entries
verify_geometry_contract = adapter.verify_geometry_contract

def backend():
    return b

class PersistentRenderBatch(FastRenderBatch):

    def __init__(self, plan_ref, entries, *, handoff_ref, execution_policy_ref, expected_plan=None, gpu_uuid=None):
        super().__init__(plan_ref, entries, handoff_ref=handoff_ref, expected_plan=expected_plan, gpu_uuid=gpu_uuid)
        self.pool = None
        self.execution_policy_ref = deepcopy(execution_policy_ref)
        self._RenderBatch__context.track_reference(execution_policy_ref, self._RenderBatch__token)
        from .action4_execution_authority import readref
        policy=readref(ROOT,execution_policy_ref)
        for reference in policy['implementation_refs']:
            self._RenderBatch__context.track_reference(reference,self._RenderBatch__token)
        for module in (__import__(__name__, fromlist=['*']), __import__('data.action4_persistent_base', fromlist=['*']), __import__('data.action4_runtime_memo', fromlist=['*'])):
            path = Path(module.__file__).resolve()
            self._RenderBatch__context.track_reference({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, self._RenderBatch__token)

    def __enter__(self):
        super().__enter__()
        try:
            self.pool = PersistentWearingBase(self._RenderBatch__sampling)
        except BaseException as exc:
            self.poison(exc)
            raise
        return self

    def render_entry(self, entry, output_dir):
        self._healthy()
        if self._RenderBatch__scope is not None:
            self.poison('Reentrant worker entry')
            raise ValueError('Non-reentrant worker entry required')
        try:
            e = entry_identity(entry)
            sid = e['sample_id']
            if self._RenderBatch__entries.get(sid) != e or sid in self._RenderBatch__attempted:
                raise ValueError('Unselected or repeated worker entry')
            self._RenderBatch__attempted.add(sid)
            row, p, binding = self._RenderBatch__context.worker_entry_inputs(e, self._RenderBatch__token)
            scope = FastEntryScope(self, e, row, p, binding)
            self._RenderBatch__scope = scope
            parameters = scope.parameters_for(e, self._RenderBatch__plan, self._RenderBatch__sampling, self._RenderBatch__rows)
            if binding is None or binding.get('schema') != 'action4.pose_formal_binding.v2':
                raise ValueError('Explicit current composition binding required')
            record = render_composition_entry(self._RenderBatch__plan, e, output_dir, self._RenderBatch__sampling, self._RenderBatch__rows, parameters, render_scope=scope, scene_pool=self.pool)
            scope.after_scene()
            self._RenderBatch__done.add(sid)
            return record
        except BaseException as exc:
            self.poison(exc)
            raise
        finally:
            self._RenderBatch__scope = None

    def close(self):
        restoration = None
        try:
            if self.pool is not None and (not self.pool.closed):
                restoration = self.pool.close()
        except BaseException as exc:
            self.poison(exc)
            raise
        receipt = super().close()
        receipt['persistent_base_restoration'] = restoration
        receipt['execution_policy_ref'] = deepcopy(self.execution_policy_ref)
        self._RenderBatch__receipt = deepcopy(receipt)
        return self.receipt

class FastEntryScope:

    def __init__(self, batch, entry, row, parameters, binding):
        if type(batch) is not PersistentRenderBatch:
            raise ValueError('Concrete live RenderBatch owner required')
        self.__batch = batch
        self.__entry = deepcopy(entry)
        self.__row = deepcopy(row)
        self.__parameters = deepcopy(parameters)
        self.__binding = deepcopy(binding)

    def _guard(self):
        return self.__batch._check_scope(self)

    def parameters_for(self, entry, plan, context, rows):
        self._guard()
        self.__batch._check_inputs(entry, plan, context, rows, self.__entry)
        if self.__parameters.get('schema') == 'action4.modular_pose_controls.v1':
            return deepcopy(self.__parameters)
        result = parameters_for(entry, context, rows)
        if result != self.__parameters:
            raise ValueError('Worker typed parameters differ from frozen bytes')
        return result

    def pre_render_binding(self, session, entry, parameters, row, plan):
        self._guard()
        if entry != self.__entry or row != self.__row or parameters != self.__parameters or (self.__binding is None):
            raise ValueError('Exact worker pose pre-render scope required')
        self.__batch._check_pre_render(session, plan)
        self.after_scene()
        return self.__batch._plan_ref_for_scope(self)

    def render_settings(self):
        self._guard()
        return self.__batch._settings_for_scope(self)

    def validate_record(self, record, entry):
        self._guard()
        if entry != self.__entry:
            raise ValueError('Worker record belongs to another entry')
        return self.__batch._record_for_scope(self, record, entry)

    def after_scene(self):
        self._guard()
        row, p, b = self.__batch._fresh_for_scope(self, self.__entry)
        if row != self.__row or p != self.__parameters or b != self.__binding:
            raise ValueError('Worker entry inputs changed during scene transaction')

def fast_pre_render(session, entry, parameters, row, plan, runtime, *, render_scope=None):
    b = backend()
    if type(render_scope) is not FastEntryScope:
        raise ValueError('Exact active FastEntryScope required')
    plan_ref = render_scope.pre_render_binding(session, entry, parameters, row, plan)
    session.verifier(session.scene)
    g = session.contract['actor_collision_diagnostics']
    counts = {side: 9 if on else 4 for side, on in zip(('left', 'right'), parameters['target_presence'])}
    if g['cross_pair_count'] != counts['left'] * counts['right'] or g['internal_pair_counts'] != {side: n * (n - 1) // 2 for side, n in counts.items()} or g['outside_or_unresolved_surface_pairs'] != 0:
        raise ValueError('All retained actual actor collision pairs required before render')
    depth = controls.verify_actual_pose_geometry(session)
    ids = [i for i, on in enumerate(parameters['target_presence'], 1) if on]
    data = runtime / 'pose_preflight_data'
    out = runtime / 'pose_preflight_runtime'
    data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    probe = pixels.render_frame(scene=session.scene, entry=deepcopy(entry), targets=session.targets(), canonical=session.contract['canonical_definition'], K=camera.actual_K(session.scene), data_out=data, runtime_out=out, output_role='diagnostic', source_identities=plan['asset_identities'], render_settings=plan['recipe']['render'], verify_scene=session.verifier, probe_only=True)
    if probe['render_calls'] != 0 or probe['render_attempts'] != 0 or probe['rendered'] is not False or (probe['restoration_verified'] is not True) or (probe['source_identities_verified'] is not True):
        raise ValueError('Pose preflight must not render')
    errors = verify_geometry_contract(parameters, session.contract, probe['geometry'], depth)
    return {'schema': 'action4.pose_formal_actual_proof.v1', 'binding_ref': deepcopy(row['pose_execution_ref']), 'plan_ref': plan_ref, 'parameters_sha256': b._digest(parameters), 'entry_sha256': b._digest(b.entry_identity(entry)), 'geometry': deepcopy(probe['geometry']), 'actual_collision': deepcopy(g), 'actual_depth': depth, 'independent_projection_error_px': errors, 'render_calls': 0, 'training_accepted': False}

def render_composition_entry(plan, entry, output_dir, context, rows, parameters, *, render_scope=None, scene_pool):
    """Formal v2 transaction: original gate -> layers -> pixels -> layers -> base.
    No enriched candidate record is published before every cleanup/source check.
    """
    from data import action4_modular_scene_sampling as sampling
    from data import action4_pose_formal_adapter as adapter
    from data import action4_formal_composition as composer
    runtime = synth._run_output_dir(output_dir)
    session = None
    record = None
    primary = None
    cleanup = []
    row = rows[entry['sample_id']]
    source_before = deepcopy(plan['asset_identities'])
    try:
        with scene_pool.apply(parameters) as session:
            proof = fast_pre_render(session, entry, parameters, row, plan, runtime, render_scope=render_scope)
            settings = deepcopy(plan['recipe']['render']) if render_scope is None else render_scope.render_settings()
            result, proof = composer.render_composition(session, entry, parameters, row, plan, runtime, proof, settings, output_dir)
            record = result['record']
            record['pose_controls_proof'] = adapter.finish_record(parameters, record, proof)
            record['asset_backend'] = BACKEND
            record['modular_backend'] = {'entry_sha256': _digest(entry_identity(entry)), 'plan_sha256': plan['_runtime_plan_sha256'], 'recipe_sha256': plan['recipe_sha256'], 'parameters_ref': row['parameters_ref'], 'parameters_sha256': _digest(parameters), 'source_context_sha256': context['source_context_sha256'], 'parameter_counts_as_formal_coverage': parameters['augmentation']['counts_as_formal_coverage'], 'base_kind': 'palmar_wearing', 'scene_contract': deepcopy(session.contract), 'pixel_receipt': str((runtime / 'samples' / entry['sample_id'] / 'render_receipt.json').relative_to(ROOT))}
    except BaseException as exc:
        primary = exc
    try:
        if session is not None and (not session.restored or scene_pool.active is not None or scene_pool.base.restored):
            raise ValueError('Persistent template restoration incomplete or base closed prematurely')
        sampling.verify_sources(context)
        if render_scope is None:
            if asset_identities(plan['recipe']) != source_before:
                raise ValueError('Composition formal source closure changed')
        else:
            render_scope.after_scene()
    except BaseException as exc:
        cleanup.append(repr(exc))
    if primary is not None or cleanup:
        synth._write(runtime / 'composition_failure.json', {'status': 'FAILED_RETAINED_NO_RETRY', 'primary': None if primary is None else repr(primary), 'cleanup': cleanup, 'training_accepted': False, 'dataset_generation_allowed': False})
        raise RuntimeError('Formal composition transaction failed: ' + repr(primary) + '; cleanup=' + repr(cleanup)) from primary
    if render_scope is None:
        validate_record(record, entry, plan['_runtime_plan_sha256'], plan, rows)
    else:
        render_scope.validate_record(record, entry)
    synth._write(runtime / 'samples' / entry['sample_id'] / 'backend_record.json', record)
    return record

def render_inside_blender(plan_path, output_dir, shard_index, num_shards, limit=None, *, gpu_uuid=None, entry_start=0, entry_count=None, handoff_ref, execution_policy_ref):
    from .action4_execution_authority import active,readref,authorize_render
    current, authority_refs=active(ROOT)
    if current.get('render',{}).get('execution_policy_ref')!=execution_policy_ref:
        raise ValueError('Exact active execution policy required')
    policy=readref(ROOT,execution_policy_ref)
    if policy.get('schema')!='action4.fast_execution_policy.v1' or policy.get('handoff_ref')!=handoff_ref:
        raise ValueError('Unapproved fast execution policy')
    for reference in policy['implementation_refs']:
        raw=_path(reference['path'],exists=True).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=reference['sha256']:raise ValueError('Fast implementation changed')
    authorize_render(ROOT,'scripts/action4_render_fast.py',plan_path,output_dir,shard_index,num_shards,entry_start,entry_count,limit)
    started=time.perf_counter()
    plan = _read(_path(plan_path, exists=True))
    if plan['recipe'].get('asset_backend') != 'modular_composed':
        raise ValueError('Fast execution supports only the approved modular plan')
    from data import action4_modular_synthesis_backend as modular
    plan_ref = {'path': str(_path(plan_path, exists=True).relative_to(ROOT)), 'sha256': _hash(plan_path)}
    entries = select_shard_entries(plan, shard_index, num_shards, limit, entry_start, entry_count)
    out = _path(output_dir, exists=True)
    run_dir = _run_output_dir(out)
    pending = out / 'manifest.worker_pending.jsonl'
    manifest_path = out / 'manifest.jsonl'
    completion_path = run_dir / 'render_completion.json'
    if any((p.exists() or p.is_symlink() for p in (pending, manifest_path, completion_path))):
        raise FileExistsError('Worker attempt artifacts already exist')
    batch = None
    publication = None
    manifest_proof = None
    manifest_digest = hashlib.sha256()
    try:
        with RuntimeMemo(plan['asset_identities']) as memo:
            with PersistentRenderBatch(plan_ref, entries, expected_plan=plan, gpu_uuid=gpu_uuid, handoff_ref=handoff_ref, execution_policy_ref=execution_policy_ref) as batch:
                for name in ('rgb', 'masks', 'scenes'):
                    (out / name).mkdir(exist_ok=False)
                run_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(synth.__file__, run_dir / 'synthesis_source_snapshot.py')
                with pending.open('x', encoding='utf-8') as manifest:
                    for entry in entries:
                        sample_started=time.perf_counter()
                        record = batch.render_entry(entry, out)
                        print('FAST_SAMPLE_SECONDS',entry['sample_id'],time.perf_counter()-sample_started,flush=True)
                        line = json.dumps(record, ensure_ascii=False) + chr(10)
                        if manifest.write(line) != len(line):
                            raise OSError('Incomplete pending worker manifest write')
                        manifest.flush()
                        manifest_digest.update(line.encode('utf-8'))
                        print('SYNTHESIS_SAMPLE_RENDERED_PENDING_BATCH_CLOSE', entry['sample_id'], len(record['instances']), flush=True)
        receipt = batch.receipt
        raw, proof = modular._ReportPublication._read_regular(pending)
        if proof['sha256'] != manifest_digest.hexdigest():
            raise ValueError('Pending manifest differs from validated worker records')
        import os
        os.link(pending, manifest_path, follow_symlinks=False)
        manifest_proof = proof
        complete = {'sample_ids': [e['sample_id'] for e in entries], 'status': 'rendered_pending_independent_validation', 'worker_batch_closing': receipt, 'manifest_ref': {'path': str(manifest_path.relative_to(ROOT)), 'sha256': proof['sha256']}, 'training_accepted': False, 'execution_policy_ref': execution_policy_ref, 'runtime_memo': memo.receipt, 'elapsed_seconds': time.perf_counter()-started}
        canonical = json.dumps(complete, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        publication = modular._ReportPublication([(completion_path, canonical)], receipt['run_id'])
        publication.publish()
        batch.receipt
        if not modular._ReportPublication._matches(manifest_path, manifest_proof):
            raise ValueError('Worker manifest changed before completion')
        cleanup = modular._ReportPublication._safe_remove(pending, proof)
        if not cleanup['removed']:
            raise RuntimeError('Owned pending manifest cleanup incomplete: ' + repr(cleanup))
    except BaseException as exc:
        if batch is not None:
            batch.poison(exc)
        failure = publication.abort(exc) if publication is not None else None
        rollback = modular._ReportPublication._safe_remove(manifest_path, manifest_proof) if manifest_proof is not None else None
        if run_dir.exists():
            import traceback
            _write(run_dir / 'worker_batch_failure.json', {'status': 'WORKER_BATCH_FAILED_NO_COMPLETION', 'error': repr(exc), 'traceback': traceback.format_exc(), 'publication_failure': failure, 'manifest_rollback': rollback, 'pending_manifest_retained': pending.exists(), 'training_accepted': False})
        raise
