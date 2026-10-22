"""Approved immutable-plan execution cache for Action4.

Reuse only exact root-authorized create_plan metadata. Fresh file identities at
both boundaries, original per-record/PNG checks, owner tokens and publication
remain enforced. Geometry, render settings, ordered8 and quotas are unchanged.
The inherited constructors pin the original backend ABI; they are not generic
adapters for arbitrary future backend versions. No scene/model work on import.
"""
from pathlib import Path
from copy import deepcopy
import os,threading,uuid,json,hashlib
from . import action4_modular_synthesis_backend as b,action4_synthesis as synth
from . import action4_execution_authority as authority
backend=b
R=synth.ROOT
entry_identity=b.entry_identity
PRODUCER_SHA='b653187fb1153437ed9b61999c7025d7d7ff47c76f16001d2a2b2eb81c76a8df'
BACKEND_SHA='b8906399efa54fe47117ed458a4487ab253ef810aaf536f174f1bd6c98529d41'

def approved_context(plan_ref,run_token,*,handoff_ref):
    current,refs=authority.active(R)
    phase=current.get('phase')
    expected=current.get('render',{}).get('plan_ref') if phase in ('render128','render_remaining') else current.get('training',{}).get('plan_ref') if phase=='training' else None
    if current.get('status')!='ACTIVE' or expected is None or plan_ref!=expected:
        raise ValueError('Exact currently root-authorized plan required for semantic reuse')
    context=_ScopedPlanContext(plan_ref,run_token,handoff_ref=handoff_ref)
    for reference in refs:context.track_reference(reference,run_token)
    path=Path(authority.__file__).resolve()
    context.track_reference({'path':str(path.relative_to(R)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},run_token)
    return context

class _FreshClosingContext(backend.ValidatedPlanContext):

    def close(self, run_token):
        self._owner_guard()
        if self._ValidatedPlanContext__closed:
            self._poison_internal('Repeated context close')
            raise ValueError('Plan context already closed')
        if run_token is not self._ValidatedPlanContext__token or os.getpid() != self._ValidatedPlanContext__pid or threading.get_ident() != self._ValidatedPlanContext__thread or (self._ValidatedPlanContext__active is not None):
            self.poison('Invalid close owner/state')
            raise ValueError('Idle same-run close required')
        self._ValidatedPlanContext__closed = True
        failures = []
        vector = dict(self._ValidatedPlanContext__index.identities)
        for path, digest in self._ValidatedPlanContext__extra_refs.items():
            if path in vector and vector[path] != digest:
                failures.append('Conflicting extra dependency: ' + path)
            vector[path] = digest
        assets = self._ValidatedPlanContext__assets
        if assets != self._ValidatedPlanContext__plan['asset_identities']:
            failures.append('Opening asset map mutated')
        for path, digest in assets.items():
            if vector.get(path) != digest:
                failures.append('Opening dependency snapshot incomplete: ' + path)
        pr = self._ValidatedPlanContext__plan_ref
        if vector.get(pr['path']) != pr['sha256']:
            failures.append('Exact plan absent from opening snapshot')
        plan_raw = None
        for path, digest in vector.items():
            try:
                raw = backend._current_reference({'path': path, 'sha256': digest})
                if path == pr['path']:
                    plan_raw = raw
            except BaseException as exc:
                failures.append(str(exc))
        try:
            if plan_raw is None or json.loads(plan_raw) != self._ValidatedPlanContext__plan:
                raise ValueError('Formal plan changed during batch')
        except BaseException as exc:
            failures.append(str(exc))
        if failures:
            self.poison(failures)
        if self._ValidatedPlanContext__poisoned:
            raise ValueError('Poisoned batch cannot close successfully; ' + repr(self._ValidatedPlanContext__errors))
        self._ValidatedPlanContext__receipt = {'status': 'PASS_RUN_LOCAL_PLAN_CLOSING_INTEGRITY_ONLY', 'run_id': self._ValidatedPlanContext__run_id, 'plan_ref': self.plan_ref, 'record_checks': self._ValidatedPlanContext__record_checks, 'opening_counts': self.opening_counts, 'closing_counts': {'reference_byte_reads': len(vector), 'dependency_graph_rebuilds': 0}, 'closing_full_reference_count': len(vector), 'dependency_map_unchanged': True, 'source_vector_sha256': backend._digest(vector), 'training_accepted': False}
        return self.receipt

class _ReadOnceIndex(b._PlanIndex):

    def check_ref(self, reference):
        self._guard()
        path = self.path(reference)
        if reference['path'] in self.identities:
            self._register(reference)
            return path
        raw = path.read_bytes()
        self.events['reference_byte_reads'] += 1
        if hashlib.sha256(raw).hexdigest() != reference['sha256']:
            raise ValueError('Opening source SHA mismatch: ' + reference['path'])
        self._register(reference)
        if path.suffix == '.json':
            self.raw_json[reference['path']] = raw
            self.documents[reference['path']] = json.loads(raw)
            self.events['JSON_parses'] += 1
        return path

    def read_ref(self, reference):
        self.check_ref(reference)
        if reference['path'] not in self.documents:
            return super().read_ref(reference)
        return self.documents[reference['path']]

class _ScopedPlanContext(_FreshClosingContext):

    def __init__(self, plan_ref, run_token, *, handoff_ref):
        if run_token is None:
            raise ValueError('Explicit run token required')
        self._ValidatedPlanContext__token = run_token
        self._ValidatedPlanContext__pid = os.getpid()
        self._ValidatedPlanContext__thread = threading.get_ident()
        self._ValidatedPlanContext__run_id = uuid.uuid4().hex
        self._ValidatedPlanContext__closed = False
        self._ValidatedPlanContext__poisoned = False
        self._ValidatedPlanContext__active = None
        self._ValidatedPlanContext__errors = []
        self._ValidatedPlanContext__extra_refs = {}
        self._ValidatedPlanContext__receipt = None
        index = _ReadOnceIndex()
        handoff = index.read_ref(handoff_ref)
        plan = index.read_ref(plan_ref)
        index.bind_plan(plan)
        if handoff.get('status') != 'ORIGINAL2000_CANDIDATE_PLAN_GEOMETRY_PENDING' or handoff.get('plan_ref') != plan_ref or handoff.get('training_accepted') is not False:
            raise ValueError('Exact original create_plan handoff required')
        if plan['count'] != 2000 or plan['status'] != 'render_candidates_pending_acceptance' or plan['recipe'].get('asset_backend') != 'modular_composed':
            raise ValueError('Current frozen2000 plan required')
        if plan['synthesis_code_sha256'] != PRODUCER_SHA or plan['asset_identities'].get('src/data/action4_modular_synthesis_backend.py') != BACKEND_SHA:
            raise ValueError('Unreviewed semantic producer version')
        if index.hash_path(synth.__file__) != PRODUCER_SHA or index.hash_path(b.__file__) != BACKEND_SHA:
            raise ValueError('Original semantic producer changed')
        rr = {'path': plan['recipe_path'], 'sha256': plan['recipe_sha256']}
        if handoff['recipe_ref'] != rr or index.read_ref(rr) != plan['recipe']:
            raise ValueError('Serialized recipe changed')
        if handoff['authority_ref'] != plan['entry_authority_ref']:
            raise ValueError('Original allocation authority mismatch')
        authority = index.read_ref(plan['entry_authority_ref'])
        if authority['entries'] != plan['entries'] or len(plan['entries']) != 2000 or len({e['sample_id'] for e in plan['entries']}) != 2000:
            raise ValueError('Complete unique frozen allocation required')
        for path, h in plan['asset_identities'].items():
            index.check_ref({'path': path, 'sha256': h})
        index.check_ref({'path': 'src/data/action4_render.py', 'sha256': plan['render_base_sha256']})
        index.check_ref(handoff['source_closing_ref'])
        spec, source, release, rows = b._release(plan['recipe'], _index=index)
        if handoff['source_context_sha256'] != release['source_context_sha256'] or b._digest(source) != release['source_context_sha256']:
            raise ValueError('Source context mismatch')
        if set(rows) != {e['sample_id'] for e in plan['entries']} or any((rows[e['sample_id']]['entry'] != e for e in plan['entries'])):
            raise ValueError('Release entry identity differs')
        context = index.sampling_context(spec, source, release)
        self._ValidatedPlanContext__index = index
        self._ValidatedPlanContext__plan_ref = deepcopy(plan_ref)
        self._ValidatedPlanContext__plan = deepcopy(plan)
        self._ValidatedPlanContext__plan_bytes = json.dumps(plan, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        self._ValidatedPlanContext__rows = b._FrozenRows(rows)
        self._ValidatedPlanContext__entries = b._FrozenRows({e['sample_id']: e for e in plan['entries']})
        self._ValidatedPlanContext__parameters = b._FrozenRows({sid: index.read_ref(row['parameters_ref']) for sid, row in rows.items()})
        self._ValidatedPlanContext__bindings = b._FrozenRows({sid: index.read_ref(row['pose_execution_ref']) for sid, row in rows.items() if 'pose_execution_ref' in row})
        self._ValidatedPlanContext__assets = dict(plan['asset_identities'])
        self._ValidatedPlanContext__record_checks = 0
        self.working_context = index.pose_context(context)
        for helper in (Path(__file__).resolve(),):
            self.track_reference({'path': str(helper.relative_to(R)), 'sha256': hashlib.sha256(helper.read_bytes()).hexdigest()}, run_token)
        self.semantic_reuse = {'kind': 'exact_current_create_plan_output', 'handoff_ref': deepcopy(handoff_ref), 'producer_sha256': PRODUCER_SHA, 'plan_ref': deepcopy(plan_ref), 'actual_scene_and_per_record_checks_not_reused': True}

    def worker_metadata(self, run_token):
        self._guard(run_token)
        return (self.working_context, self._ValidatedPlanContext__rows)

    def close(self, run_token):
        self._guard(run_token)
        self._ValidatedPlanContext__index.seal()
        receipt = super().close(run_token)
        receipt['metadata_semantic_reuse'] = deepcopy(self.semantic_reuse)
        self._ValidatedPlanContext__receipt = deepcopy(receipt)
        return self.receipt

class FastHostSession(backend.HostValidationSession):

    def __init__(self, plan_ref, *, handoff_ref):
        import os, threading
        self._HostValidationSession__pid = os.getpid()
        self._HostValidationSession__thread = threading.get_ident()
        self._HostValidationSession__token = object()
        self._HostValidationSession__context = approved_context(plan_ref, self._HostValidationSession__token, handoff_ref=handoff_ref)
        self._HostValidationSession__entered = False
        self._HostValidationSession__closed = False
        self._HostValidationSession__busy = False
        self._HostValidationSession__pending = []
        self._HostValidationSession__state = 'NEW'
        self._HostValidationSession__receipt = None
        self._HostValidationSession__failure = None

class FastRenderBatch(backend.RenderBatch):

    def __init__(self, plan_ref, entries, *, expected_plan=None, gpu_uuid=None, handoff_ref):
        import os, threading
        self._RenderBatch__pid = os.getpid()
        self._RenderBatch__thread = threading.get_ident()
        self._RenderBatch__token = object()
        self._RenderBatch__context = approved_context(plan_ref, self._RenderBatch__token, handoff_ref=handoff_ref)
        self._RenderBatch__plan = self._RenderBatch__context.plan
        self._RenderBatch__ref = self._RenderBatch__context.plan_ref
        if expected_plan is not None and {k: v for k, v in expected_plan.items() if k not in ('_runtime_plan_ref', '_runtime_plan_sha256')} != self._RenderBatch__plan:
            self._RenderBatch__context.poison('Foreign expected worker plan')
            raise ValueError('Worker batch belongs to another plan')
        self._RenderBatch__plan['_runtime_plan_ref'] = deepcopy(self._RenderBatch__ref)
        self._RenderBatch__plan['_runtime_plan_sha256'] = self._RenderBatch__ref['sha256']
        self._RenderBatch__sampling, self._RenderBatch__rows = self._RenderBatch__context.worker_metadata(self._RenderBatch__token)
        self._RenderBatch__entries = {}
        self._RenderBatch__done = set()
        self._RenderBatch__attempted = set()
        self._RenderBatch__scope = None
        self._RenderBatch__state = 'NEW'
        self._RenderBatch__receipt = None
        for entry in entries:
            e = entry_identity(entry)
            sid = e['sample_id']
            if sid in self._RenderBatch__entries:
                raise ValueError('Duplicate worker batch entry')
            row, p, b = self._RenderBatch__context.worker_entry_inputs(e, self._RenderBatch__token)
            self._RenderBatch__entries[sid] = deepcopy(e)
        self._RenderBatch__settings = deepcopy(self._RenderBatch__plan['recipe']['render'])
        if self._RenderBatch__settings['device'] == 'CPU':
            if gpu_uuid is not None:
                raise ValueError('CPU worker cannot accept GPU')
        else:
            if gpu_uuid not in self._RenderBatch__settings.get('gpu_uuids', []):
                raise ValueError('Worker GPU differs from frozen allowlist')
            self._RenderBatch__settings['worker_gpu_uuid'] = gpu_uuid
