"""CANDIDATE ONLY: explicit shared-template split protocol and release binding.

No runtime imports, model work, writes, acceptance mutation or automatic promotion.
The strict legacy background-disjoint path remains the default outside this module.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

SCHEMA = 'action4.shared_template_split_protocol.v1'
MODE = 'independent_samples_shared_scene_templates'
TEMPLATES = ('wall', 'table_full_garment', 'cloth_lift10_full_garment')
METADATA_NAMES = {'wall':'wall','table_full_garment':'table','cloth_lift10_full_garment':'cloth'}
SPLITS = {'train': 1600, 'eval': 200, 'test': 200}
TARGET_COUNTS = {'0': 54, '1': 90, '2': 1856}
SPLIT_TARGET_COUNTS = {'train': {'0':48,'1':80,'2':1472},
                       'eval': {'0':6,'1':10,'2':184}, 'test': {'0':0,'1':0,'2':200}}
CLAIM = 'Independent pose and occlusion samples within three shared fixed scene templates; no novel-background generalization claim.'
GPU3 = 'GPU-510dc520-5a9d-1317-019d-de5ed02c1b37'
OUTPUT_LAYOUT = {'train':'data/train','eval':'data/eval','test':'data/test','commit':'data/action4_dataset_commit.json'}
CRITERIA = ('canonical8_and_frame_labels', 'fixed_metric_mount', 'collision_and_actual_masks',
            'native_roi_detail_and_photometry', 'real_consumer_supervision', 'declared_shared_template_split')
LIMITS = ('partial_arm_ends', 'flat_cloth_occluders_no_grasp_simulation',
          'shared_templates_not_novel_environment', 'real_video_external_domain_separate')
OVERLAY_KEYS = {'status', 'training_accepted', 'dataset_generation_allowed',
                'candidate_record_ref', 'template_generation_release_ref', 'template_relocation_ledger_ref'}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def validate_ref(ref):
    check(isinstance(ref, dict) and set(ref) == {'path','sha256'}, 'Exact path/SHA reference required')
    name = ref['path']
    check(isinstance(name,str) and name and not name.startswith('/') and
          ':' not in name and '\\' not in name and
          all(x not in ('','.','..') for x in name.split('/')), 'Project-relative reference required')
    check(isinstance(ref['sha256'],str) and re.fullmatch('[0-9a-f]{64}',ref['sha256']) is not None,
          'Lowercase SHA256 required')
    return ref


class FileReader:
    """Explicit project root, verified reads, cache, and closing identity check."""
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.refs, self.cache = {}, {}

    def path(self, value):
        p = Path(value)
        check(not p.is_absolute() and '..' not in p.parts, 'Project-relative path required')
        p = (self.root/p).resolve(strict=True)
        check(p.is_relative_to(self.root), 'Resolved source outside project')
        return p

    def check(self, ref):
        validate_ref(ref)
        name = ref['path']
        previous = self.refs.get(name)
        check(previous is None or previous == ref['sha256'], 'Conflicting source reference')
        if previous is None:
            h = hashlib.sha256()
            with self.path(name).open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024*1024), b''):
                    h.update(chunk)
            check(h.hexdigest() == ref['sha256'], 'Source SHA differs: '+name)
            self.refs[name] = ref['sha256']
        return ref

    def document(self, ref):
        self.check(ref)
        if ref['path'] not in self.cache:
            raw = self.path(ref['path']).read_bytes()
            check(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Source changed while reading')
            self.cache[ref['path']] = ([json.loads(line) for line in raw.splitlines() if line.strip()]
                                      if ref['path'].endswith('.jsonl') else json.loads(raw))
        return self.cache[ref['path']]

    def reference(self, path):
        resolved = self.path(path)
        h = hashlib.sha256(resolved.read_bytes()).hexdigest()
        return {'path':str(resolved.relative_to(self.root)), 'sha256':h}

    def close(self):
        for name, expected in self.refs.items():
            h = hashlib.sha256()
            with self.path(name).open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024*1024), b''):
                    h.update(chunk)
            check(h.hexdigest() == expected, 'Source changed during protocol verification: '+name)


def expected_destination(record, role, instance_id):
    split = record['split']
    check(split in SPLITS, 'Unknown relocation split')
    sid = record['sample_id']
    check(isinstance(sid,str) and re.fullmatch('[A-Za-z0-9_-]{1,96}',sid), 'Safe actual sample ID required')
    if role == 'rgb':
        check(instance_id is None, 'RGB has no physical instance ID')
        return OUTPUT_LAYOUT[split]+'/images/'+sid+'.png'
    check(role in ('mask_visible','mask_amodal_expanded') and type(instance_id) is int and instance_id in (1,2),
          'Only physical IDs 1/2 have masks')
    suffix = 'visible' if role == 'mask_visible' else 'amodal_expanded'
    return OUTPUT_LAYOUT[split]+'/masks/'+sid+'_i'+str(instance_id)+'_'+suffix+'.png'


def validate_relocations(candidate_records, relocation_ledger, *, ledger_ref):
    """Exact byte-preserving destination inventory; does not publish anything."""
    check(relocation_ledger.get('schema') == 'action4.template_relocation_ledger.v1' and
          relocation_ledger.get('selection_ledger_ref') == ledger_ref and
          relocation_ledger.get('output_layout') == OUTPUT_LAYOUT,
          'Relocation ledger is not bound to selection and exact three-dir layout')
    expected = {}
    for record in candidate_records:
        sid = record['sample_id']
        entries = [('rgb',None,record['image'],record['rgb_sha256'])]
        for row in record['instances']:
            check(type(row['instance_id']) is int and row['instance_id'] in (1,2), 'Physical ID cannot be renumbered')
            for role in ('mask_visible','mask_amodal_expanded'):
                entries.append((role,row['instance_id'],row[role],row[role+'_sha256']))
        for role,ident,path,sha in entries:
            key = (sid,role,ident)
            check(key not in expected, 'Duplicate selected relocation role')
            expected[key] = {'sample_id':sid,'role':role,'instance_id':ident,
                             'source_ref':validate_ref({'path':path,'sha256':sha}),
                             'destination_path':expected_destination(record,role,ident)}
    actual, destinations = {}, set()
    for item in relocation_ledger['entries']:
        check(set(item) == {'sample_id','role','instance_id','source_ref','destination_path'},
              'Exact relocation entry fields required')
        key = (item['sample_id'],item['role'],item['instance_id'])
        check(key not in actual and item['destination_path'] not in destinations,
              'Duplicate relocation role or destination')
        check(key in expected and item == expected[key], 'Relocation does not preserve exact source bytes/physical IDs')
        actual[key] = item
        destinations.add(item['destination_path'])
    check(set(actual) == set(expected), 'Missing or extra image/mask relocations')
    return actual


def validate_commit(commit, *, commit_ref, protocol_ref, release_ref, ledger_ref,
                    relocation_ref, plan_ref):
    validate_ref(commit_ref)
    check(commit_ref['path'] == OUTPUT_LAYOUT['commit'], 'Only the final commit descriptor admits the dataset')
    check(commit.get('schema') == 'action4.template_dataset_commit.v1' and
          commit.get('status') == 'COMPLETE' and commit.get('complete') is True and
          commit.get('training_accepted') is True and commit.get('count') == 2000 and
          commit.get('split_counts') == SPLITS and commit.get('output_layout') == OUTPUT_LAYOUT,
          'Complete three-split atomic commit descriptor required')
    for key,expected in (('split_protocol_ref',protocol_ref),('release_ref',release_ref),
                         ('ledger_ref',ledger_ref),('relocation_ledger_ref',relocation_ref),('plan_ref',plan_ref)):
        check(commit.get(key) == expected, 'Commit provenance differs: '+key)
    check(set(commit.get('manifest_refs',{})) == set(SPLITS), 'Commit must bind all three manifests')
    for split,ref in commit['manifest_refs'].items():
        validate_ref(ref)
        check(ref['path'] == OUTPUT_LAYOUT[split]+'/annotations.jsonl', 'Commit manifest layout differs')
    return commit['manifest_refs']


def candidate_view(record, selected, *, release_ref=None, relocation_ref=None, relocations=None):
    if record.get('status') == 'render_candidate_not_training_accepted':
        check(record.get('training_accepted') is False and record.get('dataset_generation_allowed') is False,
              'Candidate acceptance flags differ')
        return record
    check(release_ref is not None and relocation_ref is not None and relocations is not None and
          record.get('status') == 'training_accepted' and record.get('training_accepted') is True and
          record.get('dataset_generation_allowed') is True and
          record.get('candidate_record_ref') == selected['candidate_record_ref'] and
          record.get('template_generation_release_ref') == release_ref and
          record.get('template_relocation_ledger_ref') == relocation_ref,
          'Published record lacks exact candidate/release/relocation overlay')
    restored = deepcopy({k:v for k,v in record.items() if k not in OVERLAY_KEYS})
    sid = record['sample_id']
    rgb = relocations[(sid,'rgb',None)]
    check(restored['image'] == rgb['destination_path'] and restored['rgb_sha256'] == rgb['source_ref']['sha256'],
          'Published RGB relocation differs')
    restored['image'] = rgb['source_ref']['path']
    for row in restored['instances']:
        geometry = row.get('canonical8_geometry',{})
        check(geometry.get('corner_ids') == list(range(8)) and geometry.get('instance_id') == row['instance_id'],
              'Published canonical/physical IDs cannot be renumbered')
        for role in ('mask_visible','mask_amodal_expanded'):
            check((sid,role,row['instance_id']) in relocations, 'Published physical instance is not selected')
            item = relocations[(sid,role,row['instance_id'])]
            check(row[role] == item['destination_path'] and row[role+'_sha256'] == item['source_ref']['sha256'],
                  'Published mask relocation differs')
            row[role] = item['source_ref']['path']
    restored.update(status='render_candidate_not_training_accepted',
                    training_accepted=False, dataset_generation_allowed=False)
    return restored


def validate_protocol(records, plan, protocol, selection_ledger, *, read_ref,
                      ledger_ref=None, release_ref=None, relocation_ledger=None, relocation_ledger_ref=None):
    """Validate full evidence and split design; this function never approves release."""
    check(protocol.get('schema') == SCHEMA and protocol.get('mode') == MODE,
          'Explicit versioned shared-template protocol required')
    check(protocol.get('split_counts') == SPLITS and protocol.get('target_count_counts') == TARGET_COUNTS and
          protocol.get('evaluation_claim') == CLAIM and protocol.get('novel_background_generalization') is False and
          protocol.get('external_real_video_separate') is True and protocol.get('output_layout') == OUTPUT_LAYOUT,
          'Exact counts and limited evaluation claim required')
    validate_ref(protocol['plan_ref'])
    check(read_ref(protocol['plan_ref']) == plan, 'Current plan bytes differ')
    check(plan.get('schema') == 'action4.fixed_template_batch_plan.v1' and plan.get('count') == 2000 and
          len(plan['entries']) == 2000 and plan.get('split_counts') == SPLITS,
          'Exact full 2000 plan required')
    check(selection_ledger.get('schema') == 'action4.template_assembly_evidence.v1' and
          selection_ledger.get('plan_ref') == protocol['plan_ref'], 'Ledger/current plan mismatch')
    check(read_ref(validate_ref(selection_ledger['split_protocol_ref'])) == protocol,
          'Ledger split protocol identity differs')
    if ledger_ref is not None:
        check(read_ref(validate_ref(ledger_ref)) == selection_ledger, 'Selection ledger identity differs')
    if release_ref is not None:
        check(ledger_ref is not None, 'Formal published view requires immutable ledger reference')
    producers = {canonical_hash(validate_ref(r)):r for r in selection_ledger['producing_plan_refs']}
    check(producers and len(producers) == len(selection_ledger['producing_plan_refs']),
          'Unique explicit producing plan references required')
    check(set(protocol['templates']) == set(TEMPLATES), 'Exactly three fixed templates required')
    assets = {}
    for name in TEMPLATES:
        spec = protocol['templates'][name]
        check(set(spec) == {'metadata_ref','blend_ref','baseline_parameters_ref','canonical_source_ref'},
              'Exact template asset reference fields required')
        for ref in spec.values():
            validate_ref(ref)
        meta = read_ref(spec['metadata_ref'])
        check(meta['template'] == METADATA_NAMES[name] and meta['blend_ref'] == spec['blend_ref'] and
              meta['baseline_parameters_ref'] == spec['baseline_parameters_ref'] and
              meta['canonical_definition']['source_manifest'] == spec['canonical_source_ref'],
              'Template metadata/asset identity differs')
        assets[name] = spec
    for field, path in (('policy_ref',plan['policy_path']),('layout_ref',plan['layout_path'])):
        validate_ref(protocol[field])
        check(protocol[field]['path'] == path, 'Protocol policy/layout path differs')
        read_ref(protocol[field])
    rows = selection_ledger['rows']
    check(len(records) == len(rows) == 2000, 'Exactly 2000 selected records required')
    check([r['plan_index'] for r in rows] == list(range(2000)), 'Complete ordered selection indices required')
    candidates = [read_ref(validate_ref(row['candidate_record_ref'])) for row in rows]
    relocations = None
    if relocation_ledger is not None or relocation_ledger_ref is not None:
        check(ledger_ref is not None and relocation_ledger is not None and relocation_ledger_ref is not None,
              'Immutable relocation ledger and selection references required together')
        check(read_ref(validate_ref(relocation_ledger_ref)) == relocation_ledger, 'Relocation ledger identity differs')
        relocations = validate_relocations(candidates,relocation_ledger,ledger_ref=ledger_ref)
    owners = {field:{} for field in ('sample_id','plan_index','group_id','draw','realization')}
    unique_ids, split_counts, counts = set(), Counter(), Counter()
    by_split = {s:Counter() for s in SPLITS}
    templates_by_split = {s:set() for s in SPLITS}
    for original, selected, candidate in zip(records, rows, candidates):
        i = selected['plan_index']
        planned = plan['entries'][i]
        entry, template = planned['entry'], planned['template']
        check(template in TEMPLATES and selected['template'] == template and
              selected['planned_sample_id'] == entry['sample_id'], 'Selection/current plan differs')
        check(canonical_hash(selected['producing_plan_ref']) in producers,
              'Undeclared producing plan')
        producer = read_ref(selected['producing_plan_ref'])
        check(producer.get('count') == 2000 and producer['entries'][i] == planned and
              producer['seed'] == plan['seed'], 'Producing plan entry differs from current full plan')
        record = candidate_view(original, selected, release_ref=release_ref,
                                relocation_ref=relocation_ledger_ref,relocations=relocations)
        check(record == candidate, 'Published/candidate record changed beyond allowed overlay')
        attempt = selected['generation_attempt']
        check(type(attempt) is int and 0 <= attempt < 64, 'Bounded exact attempt required')
        draw = i+attempt*producer['count']
        actual = entry['sample_id']+f'_a{attempt:02}'
        check(selected['draw_index'] == draw and selected['actual_sample_id'] == actual and
              record['sample_id'] == actual and record['planned_sample_id'] == entry['sample_id'] and
              record['plan_index'] == i and record['generation_attempt'] == attempt,
              'Chosen attempt identity differs')
        expected = dict(entry, sample_id=actual, seed=producer['seed']+draw)
        for key,value in expected.items():
            check(record.get(key) == value, 'Record/current planned entry differs: '+key)
        check(record['background_source_id'] == 'fixed_template_'+template,
              'Background identity must remain truthful and shared')
        presence = record['target_presence']
        check(isinstance(presence,list) and len(presence) == 2 and all(type(x) is bool for x in presence),
              'Exact boolean presence required')
        n = sum(presence)
        check(record['target_count'] == record['planned_target_count'] == n and
              [x['instance_id'] for x in record['instances']] == [j+1 for j,v in enumerate(presence) if v] and
              record['excluded_instances'] == [] and not any(x.get('ignore',False) for x in record['instances']),
              'Actual selected presence/support differs')
        for instance in record['instances']:
            geometry = instance.get('canonical8_geometry',{})
            check(geometry.get('corner_ids') == list(range(8)) and
                  geometry.get('instance_id') == instance['instance_id'],
                  'Canonical IDs 0..7 and physical instance IDs must stay unchanged')
        split = record['split']
        check(split in SPLITS and (split != 'test' or presence == [True,True]), 'Test must be dual target')
        proposal = read_ref(validate_ref(record['template_parameters_ref']))
        collision = read_ref(validate_ref(record['template_collision_ref']))
        execution = read_ref(validate_ref(record['template_execution_ref']))
        check(proposal['entry'] == expected and proposal['background_request']['template'] == template,
              'Proposal is not chosen actual entry')
        vector = proposal['controls']['sampling_vector']
        check(vector.get('schema') == 'action4.uniform_vector.v1' and
              vector.get('master_seed') == producer['seed'] and vector.get('sample_index') == draw,
              'Draw realization identity differs')
        check(collision.get('accepted') is True and collision.get('collision_checked') is True and
              collision.get('prohibited') == [] and collision.get('unknown') == [] and
              collision.get('status') == 'PASS_INCREMENTAL_COLLISION' and
              collision.get('sample_id') == actual and collision.get('proposal_digest') == canonical_hash(proposal) and
              collision.get('plan_index') == i and collision.get('generation_attempt') == attempt,
              'Collision receipt not bound to selected proposal')
        complete = read_ref(validate_ref(selected['complete_ref']))
        check(complete.get('status') == 'CANDIDATE_BATCH_COMPLETE_FINAL_HOST_PENDING' and
              complete['template'] == template and complete['plan_ref'] == selected['producing_plan_ref'] and
              complete['execution_sources_ref'] == record['template_execution_ref'],
              'Complete receipt is not producing batch')
        completed = [x for x in complete['entries'] if x['plan_index'] == i]
        check(len(completed) == 1 and completed[0]['record_ref'] == selected['candidate_record_ref'] and
              completed[0]['sample_id'] == actual and completed[0]['collision'] == collision,
              'Complete receipt selection mismatch')
        check(execution.get('schema') == 'action4.template_batch_sources.v1' and
              execution.get('template_loads') == 1 and execution.get('template') == template,
              'Fixed worker scene execution required')
        sources = execution['source_identities']
        for ref in (*assets[template].values(),protocol['policy_ref'],protocol['layout_ref'],selected['producing_plan_ref']):
            check(sources.get(ref['path']) == ref['sha256'], 'Frozen template source map differs')
        quality = read_ref(validate_ref(selected['quality_report_ref']))
        batch_status = quality.get('status')
        pass_batch = batch_status == 'PASS_TEMPLATE_BATCH_HOST_GPU3_CONSUMER_VISUAL_PENDING'
        reject_batch = batch_status == 'REJECT_TEMPLATE_BATCH_REPLACEMENTS_REQUIRED'
        checked_rows = quality.get('samples',[])
        check((pass_batch or reject_batch) and quality.get('test_cpu_math') is False and
              quality.get('production_quality_pass') is pass_batch and
              quality.get('complete_ref') == selected['complete_ref'] and
              quality.get('plan_ref') == selected['producing_plan_ref'] and
              quality.get('missing_planned_ids') == [] and
              quality.get('gpu_receipt',{}).get('gpu_uuid') == GPU3 and
              quality.get('expected_count') == quality.get('checked_count') == complete['count'] == len(checked_rows) == len(complete['entries']) and
              len({x['plan_index'] for x in checked_rows}) == len(checked_rows) and
              {x['plan_index'] for x in checked_rows} == {x['plan_index'] for x in complete['entries']} and
              all(x.get('status') in ('PASS','REJECT') for x in checked_rows),
              'Closed production GPU3 quality receipt missing or mismatched')
        rejected = [x['planned_sample_id'] for x in checked_rows if x['status'] == 'REJECT']
        check(quality.get('rejected_planned_ids') == rejected and bool(rejected) is reject_batch,
              'Batch semantic rejections must remain explicit')
        check(selected.get('source_batch_status') == batch_status and
              selected.get('selection_basis') == 'exact_selected_row_pass_in_closed_gpu3_batch',
              'Ledger must preserve original batch status and exact row selection basis')
        checked = [x for x in quality['samples'] if x['plan_index'] == i]
        check(len(checked) == 1 and checked[0]['status'] == 'PASS' and
              checked[0]['sample_id'] == actual and checked[0]['record_ref'] == selected['candidate_record_ref'] and
              checked[0].get('formal_dataset_rejected_unaccepted') is True,
              'Exact selected sample lacks host/GPU/consumer PASS')
        check(actual not in unique_ids, 'Duplicate actual sample ID')
        unique_ids.add(actual)
        values = {'sample_id':actual,'plan_index':i,'group_id':record['group_id'],
                  'draw':(vector['master_seed'],vector['sample_index']),
                  'realization':canonical_hash(vector['values'])}
        for field,value in values.items():
            previous = owners[field].get(value)
            check(previous is None or previous == split, 'Cross-split leakage via '+field)
            owners[field][value] = split
        split_counts[split] += 1
        counts[str(n)] += 1
        by_split[split][str(n)] += 1
        templates_by_split[split].add(template)
    check(dict(split_counts) == SPLITS and dict(counts) == TARGET_COUNTS, 'Split or target-count quotas differ')
    for split in SPLITS:
        check({str(n):by_split[split][str(n)] for n in range(3)} == SPLIT_TARGET_COUNTS[split],
              'Per-split target-count quotas differ')
        check(templates_by_split[split] == set(TEMPLATES), 'Each split must contain exactly the three declared templates')
    return {'status':'PASS_DECLARED_SHARED_TEMPLATE_PROTOCOL_NOT_FORMAL_ACCEPTANCE',
            'mode':MODE,'count':2000,'split_counts':dict(split_counts),'target_count_counts':dict(counts),
            'plan_ref':deepcopy(protocol['plan_ref']),'split_protocol_ref':deepcopy(selection_ledger['split_protocol_ref']),
            'ledger_ref':deepcopy(ledger_ref),'relocation_ledger_ref':deepcopy(relocation_ledger_ref),
            'output_layout':deepcopy(OUTPUT_LAYOUT),'evaluation_claim':CLAIM,'training_accepted':False}


def validate_release(release, protocol_ref, ledger_ref, *, read_ref, verify_ref=None):
    check(release.get('schema') == 'action4.template_generation_release.v1' and
          release.get('status') == 'FORMAL_2000_RELEASE_APPROVED', 'Separate formal release required')
    binding = release['binding']
    check(binding['split_protocol_ref'] == protocol_ref and binding['ledger_ref'] == ledger_ref,
          'Release protocol/ledger binding differs')
    check(set(release['reviews']) == {'spec','root'}, 'Both explicit root and SPEC reviews required')
    for role,ref in release['reviews'].items():
        review = read_ref(validate_ref(ref))
        check(review.get('schema') == 'action4.template_generation_review.v1' and
              review.get('role') == role and review.get('decision') == 'approve_formal_2000_release' and
              review.get('binding_sha256') == canonical_hash(binding),
              'Review not bound to exact release request')
        check(isinstance(review.get('reviewer'),str) and review['reviewer'].strip() and
              isinstance(review.get('observations'),str) and review['observations'].strip(),
              'Named reviewer and substantive observations required')
        check(set(review.get('criteria',{})) == set(CRITERIA) and
              all(review['criteria'][key] is True for key in CRITERIA), 'All review criteria required')
        check(set(LIMITS).issubset(review.get('known_limits',[])), 'Known limits must be explicit')
        evidence_map = review.get('criterion_evidence_refs',{})
        check(set(evidence_map) == set(CRITERIA), 'Each criterion needs explicit evidence references')
        for evidence_refs in evidence_map.values():
            check(isinstance(evidence_refs,list) and evidence_refs, 'Criterion evidence list cannot be empty')
            for evidence in evidence_refs:
                (verify_ref or read_ref)(validate_ref(evidence))
        if role == 'root':
            visual = review.get('reviewed_samples',[])
            check(len(visual) >= 128 and len({x['sample_id'] for x in visual}) == len(visual) and
                  all(x.get('decision') == 'accepted' and isinstance(x.get('reviewer'),str) and
                      x['reviewer'].strip() and isinstance(x.get('observations'),str) and
                      x['observations'].strip() for x in visual),
                  'Root requires at least 128 distinct actual native image reviews')
    return binding


def binding_identity(root, config, *, records_by_split=None):
    """Future admission hook: no new refs means strict legacy default."""
    data = config['data']
    protocol_ref, release_ref, commit_ref = (data.get(key) for key in
        ('split_protocol_ref','dataset_release_ref','dataset_commit_ref'))
    if protocol_ref is None and release_ref is None and commit_ref is None:
        return None
    check(all(ref is not None for ref in (protocol_ref,release_ref,commit_ref)),
          'Protocol, release and final atomic commit references are all required')
    reader = FileReader(root)
    protocol = reader.document(validate_ref(protocol_ref))
    release = reader.document(validate_ref(release_ref))
    ledger_ref = release['binding']['ledger_ref']
    binding = validate_release(release, protocol_ref, ledger_ref, read_ref=reader.document, verify_ref=reader.check)
    check(binding['plan_ref'] == protocol['plan_ref'] and binding['output_layout'] == OUTPUT_LAYOUT,
          'Release plan or exact output layout differs')
    for key in ('quota_policy_ref','assembly_implementation_ref','split_validator_ref'):
        reader.check(validate_ref(binding[key]))
    check(reader.path(binding['split_validator_ref']['path']) == Path(__file__).resolve(),
          'Release must identify the executed split validator')
    relocation_ref = validate_ref(binding['relocation_ledger_ref'])
    relocation = reader.document(relocation_ref)
    commit = reader.document(validate_ref(commit_ref))
    manifests = validate_commit(commit,commit_ref=commit_ref,protocol_ref=protocol_ref,
                                release_ref=release_ref,ledger_ref=ledger_ref,
                                relocation_ref=relocation_ref,plan_ref=protocol['plan_ref'])
    plan,ledger = reader.document(protocol['plan_ref']),reader.document(ledger_ref)
    records = []
    for split in SPLITS:
        check(data[split+'_manifest'] == manifests[split]['path'], 'Config differs from committed manifest')
        values = reader.document(manifests[split])
        check(len(values) == SPLITS[split] and all(r['split'] == split for r in values),
              'Committed split manifest count or split differs')
        check(all(r.get('status') == 'training_accepted' for r in values), 'Formal binding requires published records')
        if records_by_split is not None:
            check(records_by_split[split] == values, 'Loaded Dataset differs from committed manifest')
        records.extend(values)
    records.sort(key=lambda r:r['plan_index'])
    root_review = reader.document(release['reviews']['root'])
    by_id = {r['sample_id']:r for r in records}
    visual = root_review['reviewed_samples']
    check(all(x['sample_id'] in by_id and x['rgb_sha256'] == by_id[x['sample_id']]['rgb_sha256'] for x in visual),
          'Root visual review must refer to selected exact RGB bytes')
    check({by_id[x['sample_id']]['scene_family'] for x in visual} == {r['scene_family'] for r in records} and
          {by_id[x['sample_id']]['background_source_id'] for x in visual} == {'fixed_template_'+t for t in TEMPLATES},
          'Root native review must cover all selected families and templates')
    report = validate_protocol(records,plan,protocol,ledger,read_ref=reader.document,
                               ledger_ref=ledger_ref,release_ref=release_ref,
                               relocation_ledger=relocation,relocation_ledger_ref=relocation_ref)
    # All final files are verified against the original declared bytes; directories alone are never readiness.
    declared = {split:{'images':set(),'masks':set()} for split in SPLITS}
    for item in relocation['entries']:
        reader.check({'path':item['destination_path'],'sha256':item['source_ref']['sha256']})
        parts = Path(item['destination_path']).parts
        declared[parts[1]][parts[2]].add(parts[3])
    for split in SPLITS:
        check(not (reader.root/OUTPUT_LAYOUT[split]).is_symlink(), 'Final split must be a real directory')
        directory = reader.path(OUTPUT_LAYOUT[split])
        check({p.name for p in directory.iterdir()} == {'images','masks','annotations.jsonl'},
              'Final split directory must contain exactly images, masks and annotations')
        for role in ('images','masks'):
            check(not (reader.root/OUTPUT_LAYOUT[split]/role).is_symlink(), 'Final images/masks must be real directories')
            folder = reader.path(OUTPUT_LAYOUT[split]+'/'+role)
            check(folder.is_dir() and {p.name for p in folder.iterdir()} == declared[split][role] and
                  all(p.is_file() and not p.is_symlink() for p in folder.iterdir()),
                  'Final file layout differs from relocation inventory')
    reader.close()
    return {'mode':MODE,'split_protocol_ref':deepcopy(protocol_ref),'dataset_release_ref':deepcopy(release_ref),
            'dataset_commit_ref':deepcopy(commit_ref),'relocation_ledger_ref':deepcopy(relocation_ref),
            'ledger_ref':deepcopy(ledger_ref),'plan_ref':deepcopy(protocol['plan_ref']),
            'manifest_refs':deepcopy(manifests),'validation_sha256':canonical_hash(report),'evaluation_claim':CLAIM,
            'external_real_video_separate':True}


def check_checkpoint_binding(checkpoint, binding):
    saved = checkpoint.get('identity',{}).get('split_protocol_binding')
    check(saved == binding, 'Checkpoint split protocol/release binding changed')
    if binding is not None:
        check(checkpoint['config']['data'].get('split_protocol_ref') == binding['split_protocol_ref'] and
              checkpoint['config']['data'].get('dataset_release_ref') == binding['dataset_release_ref'] and
              checkpoint['config']['data'].get('dataset_commit_ref') == binding['dataset_commit_ref'],
              'Checkpoint config protocol/release changed')


def build_config(base_config, protocol_ref, dataset_release_ref, dataset_commit_ref, *, root):
    """Return a new config only after real formal release/manifests verify."""
    check(all(key not in base_config['data'] for key in ('split_protocol_ref','dataset_release_ref','dataset_commit_ref')),
          'Expected unchanged legacy base config')
    result = deepcopy(base_config)
    result['data']['split_protocol_ref'] = deepcopy(validate_ref(protocol_ref))
    result['data']['dataset_release_ref'] = deepcopy(validate_ref(dataset_release_ref))
    result['data']['dataset_commit_ref'] = deepcopy(validate_ref(dataset_commit_ref))
    for split in SPLITS:
        result['data'][split+'_manifest'] = OUTPUT_LAYOUT[split]+'/annotations.jsonl'
    binding_identity(root,result)
    return result
