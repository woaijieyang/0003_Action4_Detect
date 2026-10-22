"""Final host, GPU3 numerical and actual consumer checks for template batches.

No Blender import, old pose proof fabrication, acceptance mutation or CUDA at import.
CPU math is an explicit test oracle only and cannot produce a production PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import traceback

ROOT = Path('/data/zy/vln/a-test')
OUTPUT_ROOT = ROOT/'outputs/action4_diverse_sampling_v007/template_quality_v001'
GPU3_UUID = 'GPU-510dc520-5a9d-1317-019d-de5ed02c1b37'


class IntegrityError(RuntimeError):
    """Broken batch/source closure is operational failure, not sample rejection."""


def bounded(value):
    p = Path(value)
    if '..' in p.parts:
        raise IntegrityError('Path traversal forbidden')
    p = (p if p.is_absolute() else ROOT/p).resolve()
    if not p.is_relative_to(ROOT):
        raise IntegrityError('Project boundary')
    return p


def relative(value):
    return str(bounded(value).relative_to(ROOT))


def file_sha(value):
    h = hashlib.sha256()
    with bounded(value).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    with bounded(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


class Inputs:
    """Track immutable references; close all inputs once at the end of the batch."""
    def __init__(self):
        self.refs = {}
        self.json_cache = {}

    def track(self, path, expected=None):
        name = relative(path)
        if expected is not None and (not isinstance(expected, str) or
                                    re.fullmatch('[0-9a-f]{64}', expected) is None):
            raise IntegrityError('Explicit SHA256 required')
        old = self.refs.get(name)
        if old is not None:
            if expected is not None and expected != old:
                raise IntegrityError('Conflicting reference: '+name)
            return old
        actual = file_sha(name)
        if expected is not None and actual != expected:
            raise IntegrityError('Input SHA mismatch: '+name)
        self.refs[name] = actual
        return actual

    def read(self, ref):
        if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
            raise IntegrityError('Explicit path/SHA reference required')
        name = relative(ref['path'])
        self.track(name, ref['sha256'])
        if name not in self.json_cache:
            raw = bounded(name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != self.refs[name]:
                raise IntegrityError('Input changed while reading: '+name)
            self.json_cache[name] = json.loads(raw)
        return self.json_cache[name]

    def current(self, path):
        return self.read({'path': relative(path), 'sha256': self.track(path)})

    def close(self):
        for path, expected in self.refs.items():
            if file_sha(path) != expected:
                raise IntegrityError('Input changed during final checks: '+path)
        return dict(self.refs)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_integrity(condition, message):
    if not condition:
        raise IntegrityError(message)


def validate_presence(record, entry):
    presence = entry['target_presence']
    require(isinstance(presence, list) and len(presence) == 2 and
            all(type(x) is bool for x in presence), 'Exact boolean bilateral presence required')
    ids = [i+1 for i, present in enumerate(presence) if present]
    require(entry['target_count'] == len(ids), 'Entry target count differs')
    require(record['target_presence'] == presence and record['target_count'] == len(ids) and
            record['planned_target_count'] == len(ids), 'Record presence/count differs')
    require(record['excluded_instances'] == [] and
            [x['instance_id'] for x in record['instances']] == ids, 'Missing/extra/absent instance')
    require(entry['split'] != 'test' or presence == [True, True], 'Test must be dual target')
    for key, value in entry.items():
        require(record.get(key) == value, 'Record entry differs: '+key)
    require(record.get('training_accepted') is False and
            record.get('dataset_generation_allowed') is False and
            record.get('status') == 'render_candidate_not_training_accepted',
            'Only unaccepted immutable candidates allowed')
    require(not any(x.get('ignore', False) or x.get('unobservable', False)
                    for x in record['instances']), 'Unsupported target cannot pass')


def track_record_pngs(inputs, record):
    """Bind every image consumed after host validation into batch closure."""
    inputs.track(record['image'], record['rgb_sha256'])
    for row in [*record['instances'], *record['excluded_instances']]:
        for key in ('mask_visible', 'mask_amodal_expanded'):
            inputs.track(row[key], row[key+'_sha256'])


def projected_geometry(record, proposal, metadata, source, engine, semantics, data_dir):
    """Independent FP64 projection plus actual visible/amodal mask pixel hulls."""
    import numpy as np
    from scipy.spatial import ConvexHull, QhullError
    from data import action4_mask_semantics as masks
    from data.action4_adapter import DataAdapter

    DataAdapter(ROOT).validate_sample(record, check_image=False)
    require((record['width'], record['height']) == (3248, 2464), 'Native image size required')
    canonical = metadata['canonical_definition']
    require(record['canonical_source_manifest'] == canonical['source_manifest'] and
            record['corners_3d_mm'] == source['corners_3d_mm'] == canonical['corners_3d_mm'],
            'Canonical source differs')
    require(np.allclose(record['camera']['camera_to_world_m'],
                        metadata['camera_world_blender_m'], rtol=0, atol=2e-6),
            'Actual camera differs from template')
    require(record['camera']['K'] == metadata['K_native'], 'Camera K differs')
    expected_ids = {str(row['instance_id']) for row in record['instances']}
    require(set(semantics['measurements_by_instance_id']) == expected_ids, 'Mask measurement presence differs')
    for name in [record['image'], *[r[k] for r in record['instances']
                                   for k in ('mask_visible', 'mask_amodal_expanded')]]:
        require(bounded(name).is_relative_to(data_dir), 'RGB/mask outside owning batch data')
    if not record['instances']:
        require(not list((data_dir/'masks').glob(record['sample_id']+'_*')),
                'Absent targets have undeclared masks')
    results = []
    for row in record['instances']:
        ident = row['instance_id']
        g = row['canonical8_geometry']
        side = ('left', 'right')[ident-1]
        require(g['instance_id'] == ident and g['object_id'] == 1 and
                g['corner_ids'] == list(range(8)) and
                g['points_object_mm'] == source['corners_3d_mm'] and
                g['corner_order'] == source['corner_order'] and
                g['canonical_source_manifest'] == canonical['source_manifest'],
                'Immutable ordered original8 differs')
        require(g['image_size_wh'] == [record['width'], record['height']] and
                g['K_native'] == record['camera']['K'] and
                g['camera_to_world_blender_m'] == record['camera']['camera_to_world_m'] and
                g['object_to_world_m'] == row['object_to_world_m'], 'Canonical matrix binding differs')
        require(g['near_plane_m'] == metadata['near_plane_m'], 'Near plane differs')
        expected_world = np.array(metadata['W_from_capture']) @ np.array(
            proposal['actor_pose_capture_by_side'][side]['core_C_capture'])
        require(np.allclose(row['object_to_world_m'], expected_world, rtol=0, atol=3e-6),
                'Rendered actor differs from applied proposal')
        uv, depths = engine.project8(g)
        native = np.asarray(g['points_native_px'], dtype=float)
        labels = np.asarray(row['corners_px'], dtype=float)
        require(native.shape == labels.shape == (8, 2) and
                np.isfinite(native).all() and np.isfinite(labels).all() and
                np.isfinite(uv).all(), 'Nonfinite or malformed projected corners')
        require(np.allclose(uv, native, rtol=0, atol=1e-6) and
                np.allclose(uv, labels, rtol=0, atol=.003), 'Independent original8 reprojection failed')
        require(row['corners_px'] == g['legacy_corners_px'] and
                row['corner_visible'] == g['legacy_corner_visible'], 'Legacy labels changed')
        inframe = ((uv[:,0] >= -.5) & (uv[:,0] <= record['width']-.5) &
                   (uv[:,1] >= -.5) & (uv[:,1] <= record['height']-.5)).tolist()
        require(g['point_inframe'] == inframe, 'Original8 inframe flags differ')
        measured = semantics['measurements_by_instance_id'][str(ident)]
        require(not measured['ignore'] and measured['area_inframe_amodal_px'] > 0 and
                measured['area_visible_px']/measured['area_inframe_amodal_px'] >= .1 and
                measured['visible_area_at_fullinput_px'] >= 16, 'Original 10 percent/16px support failed')
        try:
            equations = ConvexHull(uv).equations
        except QhullError as exc:
            raise ValueError('Degenerate projected canonical hull') from exc
        outside = {}
        for key in ('mask_visible', 'mask_amodal_expanded'):
            array = masks._decode_mask(bounded(row[key]))
            y, x = np.nonzero(array)
            xy = np.column_stack((x, y)).astype(float)
            if key == 'mask_amodal_expanded':
                xy += np.asarray(row['amodal_expanded_origin_px'])
            outside[key] = engine.outside_hull(xy, equations)
            require(outside[key] == 0, 'Mask pixels outside original8 hull: '+str((ident, key, outside[key])))
        results.append({'instance_id': ident, 'independent_points_native_px': uv.tolist(),
                        'depth_m': depths.tolist(), 'point_inframe': inframe,
                        'independent_projection_max_error_px': float(np.max(np.abs(uv-native))),
                        'outside_hull_pixels': outside, 'points_clamped': False})
    return results


def _engine(test_cpu_math):
    if type(test_cpu_math) is not bool:
        raise ValueError('Explicit boolean test mode required')
    import torch
    torch.set_num_threads(2)
    from eval.action4_gpu_quality import GPUAid, FP64Math
    if test_cpu_math:
        return FP64Math('cpu'), None
    # GPUAid verifies visibility, physical UUID, thread ownership and no fallback.
    ref = {'path': 'src/eval/action4_gpu_quality.py',
           'sha256': file_sha('src/eval/action4_gpu_quality.py')}
    aid = GPUAid(implementation_ref=ref)
    return aid.math, aid


def result_status(rejected, missing, *, test_cpu_math):
    if test_cpu_math:
        return 'TEST_ONLY_CPU_MATH_NO_PRODUCTION_ACCEPTANCE'
    return ('REJECT_TEMPLATE_BATCH_REPLACEMENTS_REQUIRED' if rejected or missing
            else 'PASS_TEMPLATE_BATCH_HOST_GPU3_CONSUMER_VISUAL_PENDING')


def finalize_batch(complete_path, output_dir, *,
                   config_path='configs/model/action4_pilot2000_v002.json',
                   test_cpu_math=False):
    """Check one immutable completed worker batch; never mark training accepted."""
    if not __debug__:
        raise RuntimeError('Consumer assertions must remain enabled')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '2'
    if Path.cwd().resolve() != ROOT:
        raise IntegrityError('Exact project cwd required')
    out = bounded(output_dir)
    if out == OUTPUT_ROOT or not out.is_relative_to(OUTPUT_ROOT) or out.exists():
        raise IntegrityError('Fresh child directory under template_quality_v001 required')
    out.mkdir(parents=True)
    start = time.monotonic()
    results, missing = [], []
    inputs = Inputs()
    try:
        complete = inputs.current(complete_path)
        if (complete.get('status') != 'CANDIDATE_BATCH_COMPLETE_FINAL_HOST_PENDING' or
                complete.get('template_loads') != 1 or complete.get('training_accepted') is not False or
                complete.get('final_host_complete') is not False):
            raise IntegrityError('Closed unaccepted fixed-scene batch required')
        execution = inputs.read(complete['execution_sources_ref'])
        source_map = execution['source_identities']
        if execution.get('schema') != 'action4.template_batch_sources.v1' or execution.get('template_loads') != 1:
            raise IntegrityError('Template execution source closure required')
        for path, sha in source_map.items():
            inputs.track(path, sha)
        for key in ('worker_config_ref', 'plan_ref'):
            ref = complete[key]
            if source_map.get(ref['path']) != ref['sha256']:
                raise IntegrityError('Required config/plan outside frozen source map')
        worker = inputs.read(complete['worker_config_ref'])
        plan = inputs.read(complete['plan_ref'])
        from data.action4_template_batch import validate_config
        data_dir, runtime_dir = validate_config(worker, plan)
        if bounded(complete_path) != runtime_dir/'COMPLETE.json' or worker['plan_path'] != complete['plan_ref']['path']:
            raise IntegrityError('Worker completion path/plan differs')
        if complete['template'] != worker['template'] or execution['template'] != worker['template']:
            raise IntegrityError('Fixed template differs')
        from data.action4_template_runtime import TEMPLATES, validate_proposal_binding
        metadata_path = 'data/action4_scene_templates/v007_candidate/'+TEMPLATES[worker['template']]+'/template.json'
        mandatory = [metadata_path, plan['policy_path'], plan['layout_path'],
                     'src/data/action4_template_runtime.py', 'src/data/action4_parameter_sampling.py',
                     'src/data/action4_template_collision.py', 'src/data/action4_template_batch.py']
        if any(path not in source_map for path in mandatory):
            raise IntegrityError('Incomplete execution sources')
        metadata = inputs.read({'path': metadata_path, 'sha256': source_map[metadata_path]})
        for ref in (metadata['baseline_parameters_ref'], metadata['blend_ref'],
                    metadata['parent_manifest_ref'], metadata['canonical_definition']['source_manifest']):
            if source_map.get(ref['path']) != ref['sha256']:
                raise IntegrityError('Template source absent from execution closure')
        policy = inputs.read({'path': plan['policy_path'], 'sha256': source_map[plan['policy_path']]})
        anchor = inputs.read(metadata['baseline_parameters_ref'])
        canonical_source = inputs.read(metadata['canonical_definition']['source_manifest'])
        manifest_ref = complete['manifest_ref']
        if bounded(manifest_ref['path']) != data_dir/'manifest.jsonl':
            raise IntegrityError('Manifest outside worker data directory')
        inputs.track(manifest_ref['path'], manifest_ref['sha256'])
        raw = bounded(manifest_ref['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest_ref['sha256']:
            raise IntegrityError('Manifest changed while reading')
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        observed = [r.get('plan_index') for r in records]
        missing = [plan['entries'][i]['entry']['sample_id'] for i in worker['indices'] if i not in observed]
        if observed != worker['indices'] or complete['count'] != len(records) or len(complete['entries']) != len(records):
            raise IntegrityError('Missing/extra/reordered batch indices: '+str(missing))
        config = inputs.current(config_path)
        from eval.action4_synthesis_validation import CONSUMER_SOURCES, consumer_check
        for path in (*CONSUMER_SOURCES, 'src/eval/action4_synthesis_validation.py',
                     'src/data/action4_mask_semantics.py', 'scripts/action4_template_quality.py',
                     'src/eval/action4_gpu_quality.py',
                     'src/eval/action4_template_quality.py', 'src/data/action4_modular_pixels.py'):
            inputs.track(path)
        engine, aid = _engine(test_cpu_math)
        from data.action4_modular_pixels import validate_host_pngs, REQUIRED_SOURCE_PATHS
        from data import action4_parameter_sampling as sampling
        from data.action4_dataset import Action4Dataset
        for record, receipt, index in zip(records, complete['entries'], worker['indices']):
            sid = record['sample_id']
            if not isinstance(sid, str) or re.fullmatch('[A-Za-z0-9_-]{1,96}', sid) is None:
                raise IntegrityError('Unsafe sample identity')
            sample_dir = out/sid
            sample_dir.mkdir()
            expected_record_path = runtime_dir/'samples'/sid/'candidate_record.json'
            if bounded(receipt['record_ref']['path']) != expected_record_path:
                raise IntegrityError('Candidate record outside exact sample')
            if inputs.read(receipt['record_ref']) != record:
                raise IntegrityError('Manifest/candidate record mismatch')
            params_ref, collision_ref = record['template_parameters_ref'], record['template_collision_ref']
            if bounded(params_ref['path']) != expected_record_path.parent/'parameters.json' or bounded(collision_ref['path']) != expected_record_path.parent/'collision.json':
                raise IntegrityError('Parameters/collision outside exact sample')
            proposal, collision = inputs.read(params_ref), inputs.read(collision_ref)
            planned = plan['entries'][index]['entry']
            provenance = proposal['provenance']
            require_integrity(provenance['anchor_ref'] == metadata['baseline_parameters_ref'] and
                              provenance['anchor_parameter_digest'] == digest(anchor) and
                              provenance['policy_digest'] == digest(policy) and
                              proposal['canonical_source_ref'] == metadata['canonical_definition']['source_manifest'],
                              'Proposal source/digest binding differs')
            require_integrity(record['planned_sample_id'] == planned['sample_id'] and
                              record['template_execution_ref'] == complete['execution_sources_ref'],
                              'Plan/execution reference differs')
            require_integrity(set(REQUIRED_SOURCE_PATHS).issubset(record['pixel_source_identities']) and
                              all(source_map.get(path) == sha for path, sha in record['pixel_source_identities'].items()),
                              'Pixel sources differ from frozen execution map')
            require_integrity(collision == receipt['collision'] and
                              collision.get('proposal_digest') == digest(proposal),
                              'Collision/proposal immutable digest differs')
            result = {'sample_id': sid, 'planned_sample_id': planned['sample_id'], 'plan_index': index,
                      'record_ref': receipt['record_ref'], 'training_accepted': False}
            try:
                attempt = record['generation_attempt']
                require(type(attempt) is int and 0 <= attempt < worker['max_attempts'], 'Invalid generation attempt')
                expected = dict(planned, sample_id=planned['sample_id']+f'_a{attempt:02}',
                                seed=plan['seed']+index+attempt*plan['count'])
                require_integrity(plan['entries'][index]['template'] == worker['template'], 'Plan template differs')
                require(record['planned_sample_id'] == planned['sample_id'] and
                        record['template_execution_ref'] == complete['execution_sources_ref'],
                        'Plan/execution reference differs')
                validate_presence(record, expected)
                require_integrity(proposal['entry'] == expected, 'Parameter entry differs')
                require_integrity(proposal['controls']['sampling_vector'] == sampling.sample_uniform(index+attempt*plan['count'], plan['seed']), 'Sampling stream identity differs')
                validate_proposal_binding(proposal, metadata, anchor, worker['template'])
                require_integrity(receipt['sample_id'] == sid and receipt['plan_index'] == index and
                        receipt['planned_sample_id'] == planned['sample_id'] and receipt['attempt'] == attempt,
                        'Completion sample identity differs')
                require_integrity(collision == receipt['collision'] and collision.get('sample_id') == sid and
                        collision.get('proposal_digest') == digest(proposal) and
                        collision.get('plan_index') == index and collision.get('generation_attempt') == attempt,
                        'Collision receipt is not bound to exact proposal/sample')
                require(collision.get('schema') == 'action4.template_collision.v1' and
                        collision.get('accepted') is True and collision.get('collision_checked') is True and
                        collision.get('status') == 'PASS_INCREMENTAL_COLLISION' and
                        collision.get('prohibited') == [] and collision.get('unknown') == [],
                        'Collision unresolved or prohibited')
                # Integrity failures remain operational, and close covers later hull/consumer reads.
                track_record_pngs(inputs, record)
                # This is the only call that decodes/recomputes original mask semantics.
                host = validate_host_pngs(record)
                geometry = projected_geometry(record, proposal, metadata, canonical_source, engine,
                                               host['mask_semantics'], data_dir)
                # Unchanged one-row manifest, never a fabricated accepted/training record.
                consumer_manifest = sample_dir/'consumer_input.jsonl'
                with consumer_manifest.open('x') as f:
                    f.write(json.dumps(record, allow_nan=False)+'\n')
                ds = Action4Dataset(ROOT, relative(consumer_manifest), allow_diagnostic=True,
                                    split=expected['split'])
                require(ds.records == [record], 'Dataset changed source record')
                try:
                    Action4Dataset(ROOT, relative(consumer_manifest), split=expected['split'])
                except ValueError as exc:
                    if 'unaccepted' not in str(exc):
                        raise
                else:
                    raise ValueError('Formal Dataset admitted unaccepted candidate')
                consumer = consumer_check(ds.collate_fn([ds[0]])[0], config)
                require(consumer['physical_ids'] == [i+1 for i, x in enumerate(expected['target_presence']) if x],
                        'Consumer lost physical identity')
                result.update(status='PASS', host=host, geometry=geometry, consumer=consumer,
                              formal_dataset_rejected_unaccepted=True)
                if aid is not None:
                    aid.passed_records += 1
                    aid.zero_target_records += int(not record['instances'])
            except (ValueError, AssertionError) as exc:
                result.update(status='REJECT', reason=str(exc),
                              replacement_request={'same_plan_index': index, 'same_planned_entry': planned,
                              'retain_original_evidence': True, 'automatic_replacement_authorized': False})
            results.append(result)
            write_new(sample_dir/'REPORT.json', result)
        refs = inputs.close()
        receipt = (aid.receipt() if aid is not None else
                   {'schema': 'action4.test_cpu_math.v1', 'device': 'cpu', 'production_eligible': False,
                    'projection_calls': engine.projection_calls, 'hull_calls': engine.hull_calls})
        rejected = [r['planned_sample_id'] for r in results if r['status'] != 'PASS']
        report = {'schema': 'action4.template_batch_quality.v1',
                  'status': result_status(rejected, missing, test_cpu_math=test_cpu_math),
                  'complete_ref': {'path': relative(complete_path), 'sha256': inputs.refs[relative(complete_path)]},
                  'manifest_ref': manifest_ref, 'plan_ref': complete['plan_ref'],
                  'template': worker['template'], 'expected_count': len(worker['indices']),
                  'checked_count': len(results), 'pass_count': len(results)-len(rejected),
                  'rejected_planned_ids': rejected, 'missing_planned_ids': missing,
                  'samples': results, 'gpu_receipt': receipt, 'input_identities': refs,
                  'test_cpu_math': test_cpu_math, 'production_quality_pass': not test_cpu_math and not rejected and not missing,
                  'original_candidates_modified': False, 'training_accepted': False,
                  'visual_review_pending': True, 'model_predictions_tested': False,
                  'elapsed_seconds': time.monotonic()-start}
        write_new(out/'REPORT.json', report)
        return report
    except Exception as exc:
        write_new(out/'FAILED.json', {'status': 'FAILED_BATCH_NO_ACCEPTANCE',
                  'error': repr(exc), 'traceback': traceback.format_exc(),
                  'completed_sample_count': len(results), 'missing_planned_ids': missing,
                  'training_accepted': False, 'production_quality_pass': False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--complete', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default='configs/model/action4_pilot2000_v002.json')
    args = parser.parse_args(argv)
    # CPU test mode is deliberately not exposed by the production CLI.
    report = finalize_batch(args.complete, args.out, config_path=args.config)
    print(json.dumps({k: report[k] for k in ('status', 'checked_count', 'pass_count',
                                            'rejected_planned_ids', 'missing_planned_ids')}, indent=2))
    return 0 if report['production_quality_pass'] else 2
