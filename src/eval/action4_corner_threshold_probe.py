"""Finite q-only decoder experiment on frozen predictions; never promotes a model."""
import copy,difflib,hashlib,importlib.util,json,os,time
from collections import Counter
from pathlib import Path
import numpy as np
from data.action4_paths import PROJECT_ROOT as ROOT,contained_path
from eval.action4_roi_sweep import binding,read_binding

def variant_source(original,q):
    if q not in (.4,.2):raise ValueError('Only preregistered q=.4/.2')
    old='"q_corner_min": .6';schema='"schema": "action4.single_frame_corner_pose.v1"'
    if original.count(old)!=1 or original.count(schema)!=1:raise ValueError('Unexpected source policy')
    return original.replace(old,'"q_corner_min": '+str(q)).replace(schema,'"schema": "action4.single_frame_corner_pose.research_q'+str(q).replace('.','p')+'.v1"')

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def decode(module,record):
    # Deliberate whitelist: only prediction-side fields can enter decoding.
    return module.decode_single_frame_pose(**{k:record[k] for k in ('canonical_m','K_native','A224_from_native','raw_rotation','raw_translation_m','observations','route')},
        pixel_space_id=record['observations']['pixel_space_id'],raw_identity=record['roi_identity'],roi_identity=record['roi_identity'],quality_ready=False,diagnostic_candidate=True,config=module.POLICY)

def candidate_or_raw(raw,d):
    result=copy.deepcopy(raw)
    result['pose_diagnostic']={'method':'geometric_candidate_or_raw','reason':d['reason'],'geometric_candidate_accepted':d['geometric_candidate_accepted'],'deployed':False}
    if d['geometric_candidate_accepted']:
        result.update(corners_px=d['candidate_corners_native'],rotation=d['candidate_rotation'],translation_m=d['candidate_translation_m'],pose_valid=True,positive_depth_verified=True)
    return result

def compare(actual,expected,path='root'):
    if isinstance(expected,dict):
        if set(actual)!=set(expected):raise ValueError('Replay keys differ '+path)
        for k in expected:compare(actual[k],expected[k],path+'.'+k)
    elif isinstance(expected,list):
        if len(actual)!=len(expected):raise ValueError('Replay lengths differ '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,path+str(i))
    elif isinstance(expected,(float,int)) and not isinstance(expected,bool):
        if not np.isclose(actual,expected,rtol=1e-9,atol=1e-8):raise ValueError('Replay numerical mismatch '+path)
    elif actual!=expected:raise ValueError('Replay value mismatch '+path)

def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def run(eval_dir,output):
    if set(os.sched_getaffinity(0))!={58,59} or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU58-59 GPU hidden only')
    folder=contained_path(ROOT,eval_dir,exists=True);out=contained_path(ROOT,output,exists=False);out.mkdir(exist_ok=True);source=out/'source';source.mkdir(exist_ok=False)
    source_ref=binding(folder/'PREDICTION_ARTIFACT.json');scoring=read_binding(binding(folder/'scoring/SCORING_INPUTS.json'))
    if scoring['input_bindings']['predictions']!=source_ref:raise ValueError('Scoring domain prediction mismatch')
    artifact=read_binding(source_ref);results=artifact['results'];domain=scoring['domain']
    if len(results)!=650 or len(domain['prediction_units'])!=1300 or sorted(r['image_id'] for r in results)!=domain['image_ids']:raise ValueError('Full fixed eval domain required')
    frozen=contained_path(ROOT,artifact['source_manifest']['path'],exists=True).parent
    decoder_path=contained_path(ROOT,frozen/'src/model/action4_single_frame_pose.py',exists=True)
    original=decoder_path.read_text();(source/'decoder_q0p6.py').write_text(original);base=load_module(source/'decoder_q0p6.py','probe_base')
    hashes={};diffs={}
    for q in (.4,.2):
        tag='q'+str(q).replace('.','p');new=variant_source(original,q);p=source/(tag+'.py');p.write_text(new)
        mod=load_module(p,'probe_'+tag)
        changed={k for k in base.POLICY if base.POLICY[k]!=mod.POLICY[k]}
        if changed!={'schema','q_corner_min'}:raise ValueError('Non-q policy mutation')
        hashes[tag]=binding(p);diffs[tag]=''.join(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile='decoder_q0p6.py',tofile=p.name));(source/(tag+'.diff')).write_text(diffs[tag])
    inputs={n:binding(folder/n) for n in ('PREDICTION_ARTIFACT.json','RAW_REPORT.json','CANDIDATE_REPORT.json','scoring/SCORING_INPUTS.json')}
    write(out/'PREREGISTRATION.json',{'status':'FINITE_Q_ONLY_DEVELOPMENT_PROBE_NOT_PROMOTION','thresholds':[.4,.2],'baseline':.6,'minimum_ids':6,'all_other_gates_unchanged':True,'input_bindings':inputs,'annotations':scoring['input_bindings']['annotations'],'original_decoder':binding(decoder_path),'copies':hashes,'source_diff_files':[str(p.relative_to(ROOT)) for p in source.glob('*.diff')],'decode_GT_access':False,'model_calls':0,'video_threshold_unchanged':.6,'defaults_changed':False,'automatic_promotion':False,'cpu_affinity':sorted(os.sched_getaffinity(0))})
    all_obs=[o for r in results for o in r['corner_observations']]
    probes=[next(o for o in all_obs if o['decoder_diagnostics']['geometric_candidate_accepted']),next(o for o in all_obs if not o['decoder_diagnostics']['geometric_candidate_accepted'])]
    replay=[]
    for o in probes:
        d=decode(base,o);compare(d,o['decoder_diagnostics']);replay.append({'image_id':o['image_id'],'prediction_id':o['prediction_id'],'reason':d['reason'],'geometric_candidate_accepted':d['geometric_candidate_accepted'],'all_values_matched':True})
    write(out/'BASELINE_REPLAY.json',{'status':'TWO_REAL_INSTANCES_MATCH','numeric_atol':1e-8,'numeric_rtol':1e-9,'instances':replay})
    for q in (.4,.2):
        tag='q'+str(q).replace('.','p');dest=out/tag;dest.mkdir();mod=load_module(source/(tag+'.py'),'probe_run_'+tag);predictions=[];diagnostics=[];reasons=Counter();ids=Counter();start=time.monotonic()
        for image in results:
            raws={p['prediction_id']:p for p in image['raw_instances']};instances=[]
            if set(raws)!={o['prediction_id'] for o in image['corner_observations']}:raise ValueError('Instance domain differs')
            for o in image['corner_observations']:
                d=decode(mod,o);instances.append(candidate_or_raw(raws[o['prediction_id']],d));diagnostics.append({'image_id':image['image_id'],'prediction_id':o['prediction_id'],'decoder_diagnostics':d});reasons[d['reason']]+=1;ids[len(d['used_corner_ids'])]+=1
            predictions.append({'sample_id':image['image_id'],'instances':instances})
            if len(predictions)%50==0:print(json.dumps({'threshold':q,'images':len(predictions),'seconds':time.monotonic()-start}),flush=True)
        write(dest/'PREDICTIONS.json',predictions);write(dest/'DECODER_DIAGNOSTICS.json',diagnostics)
        write(dest/'DECODE_COMPLETE.json',{'images':650,'instances':len(diagnostics),'rejection_or_acceptance_reasons':dict(reasons),'used_ID_count_histogram':dict(ids),'instances_with_at_least6_IDs':sum(v for k,v in ids.items() if k>=6),'selected_ID_total':sum(k*v for k,v in ids.items()),'accepted_candidates':sum(d['decoder_diagnostics']['geometric_candidate_accepted'] for d in diagnostics),'quality_ready':False,'deployment_admitted':False,'GT_used':False,'seconds':time.monotonic()-start})
    # Predictions for BOTH candidates are frozen before any annotation is loaded.
    annotation_ref=scoring['input_bindings']['annotations'];annotation_path=contained_path(ROOT,annotation_ref['path'],exists=True)
    if binding(annotation_path)!=annotation_ref:raise ValueError('Published annotations changed')
    records=[json.loads(line) for line in annotation_path.open()];records=[r for r in records if r['sample_id'] in set(domain['image_ids'])]
    if len(records)!=650:raise ValueError('Published eval records incomplete')
    from eval.action4_pose_protocol import score_pose
    scorer_source=contained_path(ROOT,'src/eval/action4_pose_protocol.py',exists=True)
    for rel in ('src/eval/action4_pose_protocol.py','src/data/action4_instance_assignment.py'):
        if (ROOT/rel).read_bytes()!=(frozen/rel).read_bytes():raise ValueError('Scorer differs from original eval '+rel)
    comparison={}
    for name,q in [('baseline',.6),('q0p4',.4),('q0p2',.2)]:
        if name=='baseline':report=read_binding(inputs['CANDIDATE_REPORT.json']);ds=[o['decoder_diagnostics'] for o in all_obs]
        else:
            report=score_pose(json.loads((out/name/'PREDICTIONS.json').read_text()),records);write(out/name/'REPORT.json',report);ds=[r['decoder_diagnostics'] for r in json.loads((out/name/'DECODER_DIAGNOSTICS.json').read_text())]
        if report['pose_gt_count']!=1300 or report['pose_output_count']!=1300:raise ValueError('Full denominator required')
        comparison[name]={'q_corner_min':q,'All8_002':report['pose_all8_at_002_success'],'All8_005_rate':report['pose_all8_at_005_rate'],'FP':report['pose_fp'],'FN':report['pose_fn'],'P95':report['pose_corner_p95_valid_matched_only'],'groups':report['pose_groups'],'accepted_candidates':sum(d['geometric_candidate_accepted'] for d in ds),'selected_ID_total':sum(len(d['used_corner_ids']) for d in ds),'instances_with_at_least6_IDs':sum(len(d['used_corner_ids'])>=6 for d in ds),'ID_histogram':dict(Counter(len(d['used_corner_ids']) for d in ds)),'decoder_reasons':dict(Counter(d['reason'] for d in ds))}
    raw=read_binding(inputs['RAW_REPORT.json']);write(out/'COMPLETE.json',{'status':'FULL650_Q_ONLY_PROBE_COMPLETE_NOT_PROMOTION','images':650,'GT':1300,'routes':comparison,'raw_reference':{k:raw[k] for k in ('pose_all8_at_002_success','pose_all8_at_005_rate','pose_fp','pose_fn','pose_corner_p95_valid_matched_only')},'scorer':binding(scorer_source),'model_calls':0,'GPU':False,'video_threshold_unchanged':.6,'defaults_changed':False,'quality_ready':False,'deployment_admitted':False})
    print(json.dumps({'status':'COMPLETE','routes':{k:{x:v[x] for x in ('All8_002','All8_005_rate','FP','FN','P95','accepted_candidates','instances_with_at_least6_IDs')} for k,v in comparison.items()}}),flush=True)
