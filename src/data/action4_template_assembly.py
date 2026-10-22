"""Fixed-template assembly with explicit formal release gates.

Consumes existing GPU3/Host/consumer evidence without decoding PNGs or rendering.
A preview approval cannot authorize this publisher. Full root and SPEC release,
exact 2000-slot coverage and the explicit shared-template split protocol required.
"""
from copy import deepcopy
from pathlib import Path
import ctypes
import hashlib
import importlib.util
import json
import os
import uuid

ROOT = Path('/data/zy/vln/a-test')
SCHEMA = 'action4.template_assembly_evidence.v1'
GPU3 = 'GPU-510dc520-5a9d-1317-019d-de5ed02c1b37'
CRITERIA = ('canonical8_and_frame_labels', 'fixed_metric_mount',
            'collision_and_actual_masks', 'native_roi_detail_and_photometry',
            'real_consumer_supervision', 'declared_shared_template_split')
LIMITS = ('partial_arm_ends', 'flat_cloth_occluders_no_grasp_simulation',
          'shared_templates_not_novel_environment', 'real_video_external_domain_separate')
QUOTA_REF = {'path': 'configs/data/action4_pilot2000_presence_quota_v003.json',
             'sha256': '43cda95e993cf0fc4a7ee8ac958fd2167bfe25c88c1e456dd7eef0d65ec3fd05'}
OUTPUT_LAYOUT = {'train':'data/train','eval':'data/eval','test':'data/test',
                 'commit':'data/action4_dataset_commit.json'}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


class References:
    """One hash per unique file on admission, then one closing hash. No PNG decode."""
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve(strict=True)
        require(self.root.is_relative_to(ROOT), 'Project-only reference root')
        self.refs, self.documents = {}, {}
        self.closed = False

    def path(self, name, *, exists=True):
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts and str(p) == name,
                'Canonical relative project path required')
        p = self.root / p
        resolved = p.resolve(strict=exists)
        require(resolved.is_relative_to(self.root) and resolved == p,
                'Project file without symlink aliases required')
        return resolved

    @staticmethod
    def sha(path):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024*1024), b''):
                h.update(block)
        return h.hexdigest()

    def add(self, ref):
        require(not self.closed, 'Reference session already closed')
        require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'Exact path/SHA reference')
        sha = ref['sha256']
        require(isinstance(sha, str) and len(sha) == 64 and set(sha) <= set('0123456789abcdef'),
                'Lowercase SHA256 required')
        path = self.path(ref['path'])
        require(path.is_file(), 'Regular source file required')
        if ref['path'] in self.refs:
            require(self.refs[ref['path']] == sha, 'Conflicting source reference')
        else:
            require(self.sha(path) == sha, 'Source bytes differ: '+ref['path'])
            self.refs[ref['path']] = sha
        return path

    def document(self, ref):
        path = self.add(ref)
        if ref['path'] not in self.documents:
            raw = path.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'JSON changed while reading')
            self.documents[ref['path']] = json.loads(raw)
        return deepcopy(self.documents[ref['path']])

    def jsonl(self, ref):
        path = self.add(ref)
        key = ref['path']
        if key not in self.documents:
            raw = path.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Manifest changed while reading')
            self.documents[key] = [json.loads(s) for s in raw.decode().splitlines() if s.strip()]
        return deepcopy(self.documents[key])

    def add_map(self, mapping):
        require(isinstance(mapping, dict) and mapping, 'Nonempty source identities required')
        for path, sha in mapping.items():
            self.add({'path': path, 'sha256': sha})

    def close(self):
        require(not self.closed, 'Closing once only')
        for name, sha in self.refs.items():
            require(self.sha(self.path(name)) == sha, 'Closing source drift: '+name)
        self.closed = True
        return {'schema': 'action4.template_assembly_closure.v1',
                'unique_inputs': len(self.refs), 'input_identities_sha256': digest(self.refs),
                'png_decode_count': 0, 'source_closing_pass': True}


def validate_plan_ancestry(reader, plan_ref, allowed_refs):
    current = reader.document(plan_ref)
    require(current.get('schema') == 'action4.fixed_template_batch_plan.v1' and
            current.get('count') == 2000 and len(current.get('entries', [])) == 2000,
            'Exact current 2000 fixed-template plan required')
    require(current.get('split_counts') == {'train':1600,'eval':200,'test':200},
            'Exact split counts required')
    chain, seen = {}, set()
    ref = plan_ref
    while True:
        require(ref['path'] not in seen, 'Cyclic plan ancestry')
        seen.add(ref['path'])
        plan = reader.document(ref)
        chain[digest(ref)] = plan
        parent = plan.get('parent_plan_ref')
        if parent is None:
            break
        ref = parent
    require(isinstance(allowed_refs, list) and len({digest(r) for r in allowed_refs}) == len(allowed_refs),
            'Unique explicit producing plans required')
    require(allowed_refs and all(digest(r) in chain for r in allowed_refs),
            'Producing plan must be exact current plan or explicit ancestor')
    return current, {digest(r): chain[digest(r)] for r in allowed_refs}


def check_coverage(rows):
    require(len(rows)==2000 and [r['plan_index'] for r in rows]==list(range(2000)),
            'Exact ordered 2000 plan slots, no duplicates or missing slots')
    require(len({r['actual_sample_id'] for r in rows})==2000 and
            len({r['planned_sample_id'] for r in rows})==2000, 'Unique actual/planned sample identities')


def check_slot(row, record, current, producing, proposal, collision):
    """Cheap exact binding, not legacy pose reconstruction or repeated PNG geometry."""
    index = row['plan_index']
    require(type(index) is int and 0 <= index < current['count'], 'Invalid plan index')
    desired = current['entries'][index]
    require(producing['entries'][index] == desired, 'Producing/current entry or template differs')
    for key in ('count','seed','policy_path','layout_path','sampling'):
        require(producing.get(key) == current.get(key), 'Producing sampling semantics changed: '+key)
    entry = desired['entry']
    attempt = row['generation_attempt']
    require(type(attempt) is int and 0 <= attempt < 64, 'Bounded integer generation attempt required')
    draw = index + attempt*producing['count']
    actual = entry['sample_id'] + f'_a{attempt:02}'
    expected = dict(entry, sample_id=actual, seed=producing['seed']+draw)
    require(row['planned_sample_id'] == entry['sample_id'] and row['actual_sample_id'] == actual and
            row['draw_index'] == draw and row['template'] == desired['template'],
            'Selected slot/actual ID/RNG identity mismatch')
    require(record.get('planned_sample_id') == entry['sample_id'] and
            record.get('plan_index') == index and record.get('generation_attempt') == attempt and
            all(record.get(k) == v for k,v in expected.items()), 'Candidate entry differs')
    require(proposal.get('entry') == expected, 'Parameter entry differs')
    require(record.get('status') == 'render_candidate_not_training_accepted' and
            record.get('training_accepted') is False and record.get('dataset_generation_allowed') is False,
            'Only original unaccepted candidates may be admitted')
    require('candidate_record_ref' not in record and 'template_generation_release_ref' not in record and
            'template_relocation_ledger_ref' not in record,
            'Already promoted or projected record not an original candidate')
    require((record.get('width'),record.get('height')) == (3248,2464), 'Native resolution required')
    presence = expected['target_presence']
    require(isinstance(presence,list) and len(presence)==2 and all(type(x) is bool for x in presence),
            'Exact boolean presence')
    ids = [i+1 for i,on in enumerate(presence) if on]
    require(record.get('excluded_instances') == [] and
            [i['instance_id'] for i in record['instances']] == ids and
            record['target_count'] == record['planned_target_count'] == len(ids),
            'Missing/extra/ignored physical instances')
    require(expected['split'] != 'test' or ids == [1,2], 'Test must remain dual')
    require(not any(i.get('ignore') or i.get('unobservable') or i.get('status')=='unobservable'
                    for i in record['instances']), 'Ignored target cannot count')
    require(collision.get('schema')=='action4.template_collision.v1' and
            collision.get('status')=='PASS_INCREMENTAL_COLLISION' and
            collision.get('accepted') is True and collision.get('collision_checked') is True and
            collision.get('prohibited')==[] and collision.get('unknown')==[],
            'Collision not fully resolved')
    require(collision.get('sample_id')==actual and collision.get('plan_index')==index and
            collision.get('generation_attempt')==attempt and
            collision.get('proposal_digest')==digest(proposal), 'Collision proposal binding differs')
    return expected


def check_quality(report, row, record, complete):
    """Reuse only an individually successful sample, preserving mixed-batch failures."""
    require(report.get('schema')=='action4.template_batch_quality.v1' and
            report.get('status') in ('PASS_TEMPLATE_BATCH_HOST_GPU3_CONSUMER_VISUAL_PENDING',
                                     'REJECT_TEMPLATE_BATCH_REPLACEMENTS_REQUIRED') and
            report.get('test_cpu_math') is False and report.get('training_accepted') is False and
            report.get('original_candidates_modified') is False,
            'Actual completed GPU quality report required')
    require(report.get('complete_ref') == row['complete_ref'] and
            report.get('plan_ref')==row['producing_plan_ref'] and
            report.get('manifest_ref')==complete['manifest_ref'] and
            report.get('template')==row['template'], 'Quality report/batch binding differs')
    require(row.get('source_batch_status')==report['status'] and
            row.get('selection_basis')=='exact_selected_row_pass_in_closed_gpu3_batch',
            'Original batch verdict and selection basis must be explicit')
    require(report.get('missing_planned_ids')==[] and
            report.get('expected_count')==report.get('checked_count')==complete.get('count'),
            'Only complete closed quality batches may contribute samples')
    if report['status']=='REJECT_TEMPLATE_BATCH_REPLACEMENTS_REQUIRED':
        require(report.get('production_quality_pass') is False and bool(report.get('rejected_planned_ids')),
                'Semantic REJECT verdict must remain explicit and unpromoted')
    samples=report['samples']
    require(len(samples)==report['checked_count'] and
            len({x['sample_id'] for x in samples})==len(samples), 'Quality sample count/duplicates')
    passed=[x for x in samples if x.get('status')=='PASS']
    require(len(passed)==report['pass_count'], 'Quality PASS count differs')
    gpu=report.get('gpu_receipt',{})
    require(gpu.get('schema')=='action4.gpu_quality_compute.v1' and gpu.get('gpu_uuid')==GPU3 and
            gpu.get('dtype')=='torch.float64' and gpu.get('training_accepted') is False and
            gpu.get('passed_records')==len(passed), 'GPU3 actual receipt required; CPU oracle not formal')
    matches=[x for x in samples if x['sample_id']==row['actual_sample_id']]
    require(len(matches)==1, 'Missing/duplicate selected quality sample')
    result=matches[0]
    require(result.get('status')=='PASS' and result.get('record_ref')==row['candidate_record_ref'] and
            result.get('plan_index')==row['plan_index'] and
            result.get('planned_sample_id')==row['planned_sample_id'] and
            result.get('formal_dataset_rejected_unaccepted') is True and
            all(key in result for key in ('host','geometry','consumer')),
            'Selected sample lacks complete Host/geometry/consumer PASS')
    require(result['consumer'].get('physical_ids')==[x['instance_id'] for x in record['instances']],
            'Consumer identity differs')
    if report['status'].startswith('PASS_'):
        require(report.get('production_quality_pass') is True and
                report.get('rejected_planned_ids')==[] and report.get('missing_planned_ids')==[],
                'Whole-batch PASS inconsistent')
    return result


def check_release_reviews(reader, release, rows_by_actual):
    require(release.get('schema')=='action4.template_generation_release.v1' and
            release.get('status')=='FORMAL_2000_RELEASE_APPROVED',
            'Separate formal 2000 release required; preview approval cannot promote')
    reviews=release.get('reviews',{})
    require(set(reviews)=={'root','spec'} and reviews['root']!=reviews['spec'],
            'Distinct root and SPEC review references required')
    binding_sha=digest(release['binding'])
    for role,ref in reviews.items():
        review=reader.document(ref)
        require(review.get('schema')=='action4.template_generation_review.v1' and
                review.get('role')==role and review.get('decision')=='approve_formal_2000_release' and
                review.get('binding_sha256')==binding_sha and
                isinstance(review.get('reviewer'),str) and bool(review['reviewer'].strip()) and
                isinstance(review.get('observations'),str) and bool(review['observations'].strip()),
                'Explicit bound formal review required')
        require(set(review.get('criteria',{}))==set(CRITERIA) and
                all(review['criteria'][key] is True for key in CRITERIA),
                'Six truthful template engineering/consumer criteria required')
        require(set(LIMITS)<=set(review.get('known_limits',[])), 'Known synthetic scope must be disclosed')
        refs=review.get('criterion_evidence_refs',{})
        require(set(refs)==set(CRITERIA) and all(refs[k] for k in CRITERIA),
                'Every criterion needs actual source evidence refs')
        for values in refs.values():
            for evidence in values:
                reader.add(evidence)
        if role=='root':
            visual=review.get('reviewed_samples',[])
            require(len({v.get('sample_id') for v in visual})==len(visual) and len(visual)>=128,
                    'At least 128 distinct actual native visual reviews')
            families,templates=set(),set()
            for item in visual:
                require(item.get('sample_id') in rows_by_actual, 'Visual sample outside final selection')
                row,record=rows_by_actual[item['sample_id']]
                require(item.get('rgb_sha256')==record['rgb_sha256'] and
                        item.get('decision')=='accepted' and item.get('reviewer') and item.get('observations'),
                        'Visual review requires exact image and accepted observations')
                families.add(record['scene_family']);templates.add(row['template'])
            require(families=={r['scene_family'] for _,r in rows_by_actual.values()} and
                    templates=={row['template'] for row,_ in rows_by_actual.values()},
                    'Native visual review must cover all selected families/templates')


def load_bound_module(reader, ref, name):
    path=reader.add(ref)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def certify(ledger_ref, release_ref):
    """Returns verified candidates, not accepted source records; closes before publish."""
    from data.action4_adapter import DataAdapter
    from data.action4_modular_synthesis_backend import validate_formal_targets, validate_presence_quota
    from data.action4_parameter_sampling import sample_uniform
    reader=References()
    ledger=reader.document(ledger_ref);release=reader.document(release_ref)
    require(ledger.get('schema')==SCHEMA, 'Explicit template ledger schema')
    require(release.get('schema')=='action4.template_generation_release.v1' and
            release.get('status')=='FORMAL_2000_RELEASE_APPROVED', 'Formal release required')
    binding=release['binding']
    require(set(binding)=={'ledger_ref','plan_ref','split_protocol_ref','quota_policy_ref','output_layout',
                           'assembly_implementation_ref','split_validator_ref','relocation_ledger_ref'}, 'Exact formal release binding')
    require(binding['ledger_ref']==ledger_ref and binding['plan_ref']==ledger['plan_ref'] and
            binding['split_protocol_ref']==ledger['split_protocol_ref'] and
            binding['quota_policy_ref']==QUOTA_REF and binding['output_layout']==OUTPUT_LAYOUT,
            'Release/ledger/current quota/output mismatch')
    own={'path':str(Path(__file__).resolve().relative_to(ROOT)),
         'sha256':References.sha(Path(__file__).resolve())}
    require(binding['assembly_implementation_ref']==own, 'Release binds another assembly implementation')
    reader.add(own);reader.document(QUOTA_REF)
    current,allowed=validate_plan_ancestry(reader,ledger['plan_ref'],ledger['producing_plan_refs'])
    rows=ledger['rows']
    check_coverage(rows)
    by_actual={};records=[];batches={}
    adapter=DataAdapter(ROOT)
    for row in rows:
        require(set(row)=={'plan_index','planned_sample_id','actual_sample_id','generation_attempt',
                          'draw_index','template','producing_plan_ref','candidate_record_ref',
                          'complete_ref','quality_report_ref','source_batch_status','selection_basis'}, 'Exact selection row schema')
        producing=allowed.get(digest(row['producing_plan_ref']))
        require(producing is not None, 'Unapproved producing ancestor')
        key=digest([row['complete_ref'],row['quality_report_ref']])
        if key not in batches:
            complete=reader.document(row['complete_ref']);quality=reader.document(row['quality_report_ref'])
            require(complete.get('status')=='CANDIDATE_BATCH_COMPLETE_FINAL_HOST_PENDING' and
                    complete.get('template_loads')==1 and complete.get('training_accepted') is False and
                    complete.get('final_host_complete') is False and
                    complete.get('plan_ref')==row['producing_plan_ref'] and
                    complete.get('template')==row['template'], 'Original completed batch required')
            execution=reader.document(complete['execution_sources_ref'])
            require(execution.get('schema')=='action4.template_batch_sources.v1' and
                    execution.get('template_loads')==1 and execution.get('template')==row['template'],
                    'Actual template execution source closure')
            reader.add_map(execution['source_identities'])
            require(all(path in quality['input_identities'] for path in (
                'src/eval/action4_template_quality.py','src/eval/action4_gpu_quality.py',
                'src/data/action4_dataset.py')), 'Missing actual quality/consumer implementation closure')
            reader.add_map(quality['input_identities'])
            worker=reader.document(complete['worker_config_ref'])
            require(worker['plan_path']==row['producing_plan_ref']['path'] and
                    worker['template']==row['template'], 'Worker producing plan/template mismatch')
            manifest=reader.jsonl(complete['manifest_ref'])
            require(len(manifest)==complete['count'] and
                    len({r['sample_id'] for r in manifest})==len(manifest), 'Closed manifest count/duplicates')
            batches[key]=(complete,quality,worker,{r['sample_id']:r for r in manifest})
        complete,quality,worker,manifest=batches[key]
        record=reader.document(row['candidate_record_ref'])
        require(manifest.get(row['actual_sample_id'])==record, 'Candidate bytes not same as original manifest row')
        require(record['template_execution_ref']==complete['execution_sources_ref'], 'Execution ref mismatch')
        require(row['plan_index'] in worker['indices'] and
                row['generation_attempt']<worker['max_attempts'], 'Unlaunched index/attempt')
        selected=[x for x in complete['entries'] if x['sample_id']==row['actual_sample_id']]
        require(len(selected)==1, 'Missing or duplicate completion entry')
        done=selected[0]
        require(done.get('record_ref')==row['candidate_record_ref'] and
                done.get('plan_index')==row['plan_index'] and
                done.get('planned_sample_id')==row['planned_sample_id'] and
                done.get('attempt')==row['generation_attempt'] and
                done.get('draw_index')==row['draw_index'], 'Completion entry mismatch')
        proposal=reader.document(record['template_parameters_ref'])
        collision=reader.document(record['template_collision_ref'])
        require(collision==done['collision'], 'Collision receipt differs from completed batch')
        for required_ref in (row['complete_ref'],row['candidate_record_ref'],row['producing_plan_ref'],
                             complete['manifest_ref'],complete['execution_sources_ref'],
                             record['template_parameters_ref'],record['template_collision_ref']):
            require(quality['input_identities'].get(required_ref['path'])==required_ref['sha256'],
                    'Required evidence absent from quality input closure')
        expected=check_slot(row,record,current,producing,proposal,collision)
        require(proposal['controls']['sampling_vector']==sample_uniform(row['draw_index'],producing['seed']),
                'Actual independent RNG vector mismatch')
        check_quality(quality,row,record,complete)
        # Check every actual PNG identity occurs in the independently closed quality inputs.
        image_refs={record['image']:record['rgb_sha256']}
        for instance in record['instances']:
            for field in ('mask_visible','mask_amodal_expanded'):
                image_refs[instance[field]]=instance[field+'_sha256']
        require(all(quality['input_identities'].get(k)==v for k,v in image_refs.items()),
                'Missing actual RGB/mask byte evidence')
        reader.add_map(image_refs)
        adapter.validate_sample(record,check_image=False)
        validate_formal_targets(record,expected,{'minimum_visible_fraction':.1,
                                                'minimum_visible_fullinput_area_px':16})
        by_actual[record['sample_id']]=(row,record);records.append(record)
    quota_view={'count':current['count'],'entries':[x['entry'] for x in current['entries']]}
    validate_presence_quota(quota_view,QUOTA_REF)
    protocol=reader.document(ledger['split_protocol_ref'])
    validator=load_bound_module(reader,binding['split_validator_ref'],'template_split_protocol_candidate')
    split_result=validator.validate_protocol(records,current,protocol,ledger,
                         read_ref=reader.document,ledger_ref=ledger_ref)
    relocation=reader.document(binding['relocation_ledger_ref'])
    validate_relocations(relocation,records,ledger_ref,binding['output_layout'])
    check_release_reviews(reader,release,by_actual)
    return reader,records,{'plan_ref':ledger['plan_ref'],'ledger_ref':ledger_ref,
                          'release_ref':release_ref,'split_protocol_ref':ledger['split_protocol_ref'],
                          'split_validation':split_result,'output_layout':binding['output_layout'],
                          'relocation_ledger_ref':binding['relocation_ledger_ref'],
                          'count':2000,'split_counts':current['split_counts'],
                          'source_quality_batches':[{'complete_ref':q['complete_ref'],
                              'status':q['status'],'production_quality_pass':q['production_quality_pass'],
                              'rejected_planned_ids':q['rejected_planned_ids']} for _,q,_,_ in batches.values()]}


def relocation_entries_for(record, layout):
    """Deterministic paths are explicit release evidence, never implicit renaming."""
    import re
    sid,split=record['sample_id'],record['split']
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,96}',sid) is not None and split in ('train','eval','test'),
            'Safe sample ID and split required for relocation')
    base=layout[split]
    rows=[{'sample_id':sid,'role':'rgb','instance_id':None,
           'source_ref':{'path':record['image'],'sha256':record['rgb_sha256']},
           'destination_path':base+'/images/'+sid+'.png'}]
    for item in record['instances']:
        ident=item['instance_id']
        require(type(ident) is int and ident in (1,2),'Physical instance ID required')
        for role,suffix in (('mask_visible','visible'),('mask_amodal_expanded','amodal_expanded')):
            rows.append({'sample_id':sid,'role':role,'instance_id':ident,
                         'source_ref':{'path':item[role],'sha256':item[role+'_sha256']},
                         'destination_path':base+'/masks/'+sid+f'_i{ident}_'+suffix+'.png'})
    return rows


def validate_relocations(relocation, records, ledger_ref, layout):
    require(relocation.get('schema')=='action4.template_relocation_ledger.v1' and
            relocation.get('selection_ledger_ref')==ledger_ref and
            relocation.get('output_layout')==layout, 'Relocation ledger binding differs')
    expected=[entry for record in records for entry in relocation_entries_for(record,layout)]
    require(relocation.get('entries')==expected,
            'Exact complete ordered pixel relocation required; no ID/byte relabeling')
    require(len({e['destination_path'] for e in expected})==len(expected),
            'Destination collision in relocation ledger')
    by_sample={}
    for entry in expected:
        by_sample.setdefault(entry['sample_id'],[]).append(entry)
    return by_sample


def accepted_copy(candidate,row,release_ref,*,relocation_rows,relocation_ref,layout):
    require(relocation_rows==relocation_entries_for(candidate,layout),
            'Admission relocation differs from original IDs/PNG hashes')
    result=deepcopy(candidate)
    require(candidate.get('status')=='render_candidate_not_training_accepted' and
            candidate.get('training_accepted') is False and candidate.get('dataset_generation_allowed') is False,
            'Admission cannot rewrite an already accepted source')
    mapping={(e['role'],e['instance_id']):e['destination_path'] for e in relocation_rows}
    result['image']=mapping['rgb',None]
    for item in result['instances']:
        for role in ('mask_visible','mask_amodal_expanded'):
            item[role]=mapping[role,item['instance_id']]
    result.update(status='training_accepted',training_accepted=True,dataset_generation_allowed=True,
                  candidate_record_ref=deepcopy(row['candidate_record_ref']),
                  template_generation_release_ref=deepcopy(release_ref),
                  template_relocation_ledger_ref=deepcopy(relocation_ref))
    return result


def _rename_no_replace(source, destination):
    source,destination=Path(source),Path(destination)
    require(source.resolve()==source and not source.is_symlink() and source.is_relative_to(ROOT) and
            destination.parent.resolve()==destination.parent and destination.parent.is_relative_to(ROOT),
            'Exact project publication paths required')
    require(source.stat().st_dev==destination.parent.stat().st_dev,'Atomic publication needs one filesystem')
    libc=ctypes.CDLL(None,use_errno=True);rename=getattr(libc,'renameat2',None)
    require(rename is not None,'No safe no-replace publication support')
    rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    rename.restype=ctypes.c_int
    if rename(-100,os.fsencode(source),-100,os.fsencode(destination),1)!=0:
        code=ctypes.get_errno();raise OSError(code,os.strerror(code),str(destination))


def materialize_pixel(reader, entry, destination):
    """Hardlinks preserve immutable bytes; verified copy only when links unsupported."""
    import errno
    import shutil
    source=reader.add(entry['source_ref'])
    destination=Path(destination)
    require(destination.is_relative_to(ROOT) and destination.parent.resolve()==destination.parent and
            not destination.exists() and not destination.is_symlink(), 'Fresh exact pixel destination required')
    try:
        os.link(source,destination)
        before,after=source.stat(),destination.stat()
        require((before.st_dev,before.st_ino)==(after.st_dev,after.st_ino),'Hardlink inode identity differs')
        mode='hardlink'
    except OSError as exc:
        supported={errno.EXDEV,errno.EPERM,errno.EOPNOTSUPP}
        if exc.errno not in supported:
            raise
        # Exclusive destination: never replace an existing or foreign file.
        with source.open('rb') as src,destination.open('xb') as dst:
            shutil.copyfileobj(src,dst,1024*1024)
        require(References.sha(destination)==entry['source_ref']['sha256'],'Copied PNG bytes differ')
        mode='verified_copy'
    return mode


def seal_split_staging(stage,invocation_id,expected_files):
    """Ownership receipt is for precise rollback, not an acceptance authority."""
    stage=Path(stage)
    require(stage.is_dir() and stage.resolve()==stage and stage.is_relative_to(ROOT),'Exact owned stage')
    found={str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file() or p.is_symlink()}
    require(found==set(expected_files),'Staged file set differs')
    require((stage/'images').is_dir() and (stage/'masks').is_dir() and
            'annotations.jsonl' in expected_files,'Complete split staging required')
    inventory={}
    directories={}
    for p in stage.rglob('*'):
        require(not p.is_symlink() and p.resolve().is_relative_to(stage),'Staged path aliases prohibited')
        if p.is_dir():
            info=p.stat();directories[str(p.relative_to(stage))]={'device':info.st_dev,'inode':info.st_ino}
    for name,sha in expected_files.items():
        p=stage/name;stat=p.stat()
        inventory[name]={'sha256':sha,'device':stat.st_dev,'inode':stat.st_ino,'size':stat.st_size}
    directory=stage.stat()
    owner={'schema':'action4.publication_ownership.v1','invocation_id':invocation_id,
           'files':inventory,'directories':directories,
           'directory_identity':{'device':directory.st_dev,'inode':directory.st_ino}}
    # Receipt stays in owned outputs; final split contains only the user layout.
    with (stage.parent/('ownership_'+stage.name+'_'+invocation_id+'.json')).open('x') as f:
        json.dump(owner,f,sort_keys=True)
    return owner


def owned_split_matches(path,owner,*,verify_bytes):
    """Foreign additions/replacements preserve the entire directory on rollback."""
    path=Path(path)
    if not path.is_dir() or path.is_symlink() or path.resolve()!=path or not path.is_relative_to(ROOT):
        return False
    try:
        top=path.stat()
        if owner.get('directory_identity')!={'device':top.st_dev,'inode':top.st_ino}:
            return False
        files=set();directories={}
        for p in path.rglob('*'):
            if p.is_symlink() or not p.resolve().is_relative_to(path):
                return False
            if p.is_dir():
                info=p.stat();directories[str(p.relative_to(path))]={'device':info.st_dev,'inode':info.st_ino}
            elif p.is_file():files.add(str(p.relative_to(path)))
            else:return False
        if files!=set(owner['files']) or directories!=owner['directories']:
            return False
        for name,value in owner['files'].items():
            p=path/name;stat=p.stat()
            if (stat.st_dev,stat.st_ino,stat.st_size)!=(value['device'],value['inode'],value['size']):
                return False
            if verify_bytes and References.sha(p)!=value['sha256']:
                return False
        return True
    except (OSError,ValueError,KeyError):
        return False


def rollback_owned_split(path,owner):
    """Called only for directories this invocation actually renamed successfully."""
    path=Path(path)
    if not owned_split_matches(path,owner,verify_bytes=True):
        return 'PRESERVED_FOREIGN_OR_CHANGED_CONTENT'
    # Exact inventory already proves ownership. Unlink never changes raw hardlink bytes.
    for name in owner['files']:
        (path/name).unlink()
    for name in sorted(owner['directories'],key=lambda x:len(Path(x).parts),reverse=True):
        (path/name).rmdir()
    path.rmdir()
    return 'ROLLED_BACK_EXACT_OWNED_DIRECTORY'


def validate_commit_shape(commit):
    require(commit.get('schema')=='action4.template_dataset_commit.v1' and
            commit.get('status')=='COMPLETE' and commit.get('complete') is True and
            commit.get('training_accepted') is True and
            set(commit.get('manifest_refs',{}))=={'train','eval','test'},
            'Complete final dataset commit required; split directory presence is insufficient')


def publish_dataset(staging, destinations, commit_path, commit, owners):
    """Three no-replace directory moves; only final descriptor is globally visible commit.

    The moves are NOT globally atomic. A crash may leave split directories but
    no descriptor; every consumer must reject that state. Graceful failure rolls
    back only exact, invocation-owned inventories and preserves foreign changes.
    """
    staging,commit_path=Path(staging),Path(commit_path)
    destinations={s:Path(p) for s,p in destinations.items()}
    validate_commit_shape(commit)
    require(set(destinations)==set(owners)=={'train','eval','test'},'Exactly three split destinations')
    for path in (*destinations.values(),commit_path):
        require(path.is_relative_to(ROOT) and path.parent.resolve()==path.parent and
                not path.exists() and not path.is_symlink(),'Existing target refused before any publication')
    for split,path in destinations.items():
        require(owned_split_matches(staging/split,owners[split],verify_bytes=False),'Staging changed before publication')
        require(commit['manifest_refs'][split]=={
            'path':str((path/'annotations.jsonl').relative_to(ROOT)),
            'sha256':References.sha(staging/split/'annotations.jsonl')},'Commit manifest identity differs')
    pending=staging/'dataset_commit.pending.json'
    with pending.open('x') as f:
        json.dump(commit,f,indent=2,allow_nan=False)
        f.flush();os.fsync(f.fileno())
    published=[]
    try:
        for split in ('train','eval','test'):
            _rename_no_replace(staging/split,destinations[split])
            published.append(split)
        # Final destination-byte closure covers copied files as well as hardlinks.
        # Same-inode/same-size mutations must not reach the single final commit.
        for split,path in destinations.items():
            require(owned_split_matches(path,owners[split],verify_bytes=True),'Published split has foreign changes')
            require(References.sha(path/'annotations.jsonl')==commit['manifest_refs'][split]['sha256'],
                    'Published manifest changed before commit')
        _rename_no_replace(pending,commit_path)
    except BaseException as exc:
        rollback={}
        if not commit_path.exists():
            for split in reversed(published):
                rollback[split]=rollback_owned_split(destinations[split],owners[split])
        failure={'status':'PUBLICATION_FAILED_NO_GLOBAL_COMMIT','error':repr(exc),
                 'commit_exists':commit_path.exists(),'rollback':rollback}
        failure_path=staging/('publication_failure_'+commit['invocation_id']+'.json')
        with failure_path.open('x') as f:
            json.dump(failure,f,indent=2)
        raise
    return {'commit_path':str(commit_path.relative_to(ROOT)),
            'commit_sha256':References.sha(commit_path),'split_renames_globally_atomic':False}


def assemble(ledger_ref, release_ref):
    """Formal publisher; never called by draft tests or live producer."""
    reader,records,summary=certify(ledger_ref,release_ref)
    layout=summary['output_layout']
    destinations={s:reader.path(layout[s],exists=False) for s in ('train','eval','test')}
    commit_path=reader.path(layout['commit'],exists=False)
    require(not any(p.exists() or p.is_symlink() for p in (*destinations.values(),commit_path)),
            'Existing formal target refused; never overwrite')
    invocation=uuid.uuid4().hex
    work=ROOT/'outputs/action4_template_assembly_runs'/invocation
    require(work.resolve().is_relative_to(ROOT/'outputs'),'Staging project boundary')
    work.mkdir(parents=True,exist_ok=False);staging=work/'dataset_pending';staging.mkdir()
    ledger=reader.document(ledger_ref)
    relocation=reader.document(summary['relocation_ledger_ref'])
    mapping=validate_relocations(relocation,records,ledger_ref,layout)
    accepted=[accepted_copy(r,row,release_ref,relocation_rows=mapping[r['sample_id']],
              relocation_ref=summary['relocation_ledger_ref'],layout=layout)
              for r,row in zip(records,ledger['rows'])]
    expected={s:{} for s in destinations};methods={'hardlink':0,'verified_copy':0}
    for split in destinations:
        (staging/split/'images').mkdir(parents=True);(staging/split/'masks').mkdir()
    for source,derived in zip(records,accepted):
        split=source['split']
        for entry in mapping[source['sample_id']]:
            suffix=Path(entry['destination_path']).relative_to(layout[split])
            methods[materialize_pixel(reader,entry,staging/split/suffix)]+=1
            expected[split][str(suffix)]=entry['source_ref']['sha256']
    from data.action4_adapter import DataAdapter
    adapter=DataAdapter(ROOT)
    for split in destinations:
        target=staging/split/'annotations.jsonl'
        with target.open('x') as f:
            for record in accepted:
                if record['split']==split:
                    adapter.validate_sample(record,check_image=False)
                    f.write(json.dumps(record,allow_nan=False)+'\n')
        expected[split]['annotations.jsonl']=References.sha(target)
    owners={s:seal_split_staging(staging/s,invocation,expected[s]) for s in destinations}
    manifest_refs={s:{'path':layout[s]+'/annotations.jsonl',
                      'sha256':expected[s]['annotations.jsonl']} for s in destinations}
    # Closing checks original paths before any trainer/gate/runner patch is applied.
    closure=reader.close()
    with (work/'assembly_closure.json').open('x') as f:
        json.dump({'closure':closure,'materialization':methods},f,indent=2)
    commit={**summary,'schema':'action4.template_dataset_commit.v1','status':'COMPLETE',
            'complete':True,'training_accepted':True,'manifest_refs':manifest_refs,
            'invocation_id':invocation,'materialization':methods,
            'claim_scope':'independent samples under three shared fixed templates; not unseen environments'}
    publication=publish_dataset(staging,destinations,commit_path,commit,owners)
    return {'commit':commit,'publication':publication}

