"""GPU3 streaming quality draft. Explicit known shards only; no generator control.
Run only after bound root128 acceptance. Importing this module does not use CUDA.
"""
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib,json,multiprocessing,os,re,time,traceback
from eval import action4_synthesis_validation as base
from eval import action4_corner_checks as original

ROOT=Path('/data/zy/vln/a-test')
GPU3_UUID='GPU-510dc520-5a9d-1317-019d-de5ed02c1b37'
ACCEPTED128={'path':'outputs/action4_fast_pipeline_repair_v001/ROOT_ACCEPTED128.json','sha256':'146a5d4a1b09d4f5604ccb795da79e134fe7db6acb663a8330a25cafed116880'}
BASE_REF={'path':'src/eval/action4_synthesis_validation.py','sha256':'24118446f348ceea3b227a7ef59520cb26db6e9cb0ad8b5072778cdfef5ac8bd'}
bounded,ref,readref,write_new=base.bounded,base.ref,base.readref,base.write_new
consumer_check=base.consumer_check


def initialize_worker():
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='2'
    os.environ['CUDA_VISIBLE_DEVICES']=GPU3_UUID
    import torch
    torch.set_num_threads(2)


def gpu_aid(reference):
    # Installation target is explicit; no dynamic imports or module patching.
    from eval.action4_gpu_quality import GPUAid
    return GPUAid(implementation_ref=reference)


def rejection(entry,manifest_ref,error):
    return {'status':'REJECT_REVIEW_AND_REPLACEMENT_REQUEST','expected_entry':deepcopy(entry),
            'manifest_ref':deepcopy(manifest_ref),'reason':str(error),
            'required_same_split':entry['split'],'required_same_group_id':entry['group_id'],
            'required_same_presence':deepcopy(entry['target_presence']),
            'original_evidence_must_be_retained':True,
            'replacement_requires_root_authorized_attempt_or_plan_revision':True,
            'automatic_acceptance':False,'training_accepted':False}


def verify_code(module_ref,quality_ref,consumers):
    for expected in [module_ref,quality_ref,BASE_REF,base.AID_REF,*consumers]:
        if ref(expected['path'])!=expected:raise ValueError('Validator/consumer implementation drift: '+expected['path'])
    if ref(__file__)!=module_ref:raise ValueError('Executing stream differs from configured module')


def snapshot(reader,plan,plan_ref,shard):
    """Original input_snapshot gates, scoped to one explicit configured partition."""
    from data.action4_synthesis import select_shard_entries,_run_output_dir
    data=bounded(shard['data_dir']);run=bounded(_run_output_dir(data))
    if not data.is_relative_to(ROOT/'data'):raise ValueError('Project data shard required')
    launch=reader.json(run/'launch.json');completion=reader.json(run/'render_completion.json')
    host=reader.json(run/'geometry_validation.json');timing=reader.json(run/'fast_launch_timing.json')
    raw=reader.data(data/'manifest.jsonl');rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    mref={'path':str((data/'manifest.jsonl').relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest()}
    command=launch['command']
    if launch.get('plan_sha256')!=plan_ref['sha256'] or original.rel(original.command_value(command,'--plan',''))!=plan_ref['path'] or original.rel(original.command_value(command,'--output',''))!=str(data.relative_to(ROOT)):
        raise ValueError('Launch plan/output mismatch')
    si=int(original.command_value(command,'--shard-index','0'));ns=int(original.command_value(command,'--num-shards','1'));start=int(original.command_value(command,'--entry-start','0'))
    n=original.command_value(command,'--entry-count');limit=original.command_value(command,'--limit')
    if ns<1 or not 0<=si<ns or start<0 or n is not None and limit is not None:raise ValueError('Invalid launch partition')
    n=int(n if n is not None else limit) if n is not None or limit is not None else None
    chosen=select_shard_entries(plan,si,ns)[start:None if n is None else start+n]
    ids=[e['sample_id'] for e in chosen]
    if not ids or ids!=shard['expected_sample_ids'] or [r['sample_id'] for r in rows]!=ids or launch['expected_sample_ids']!=ids or completion['sample_ids']!=ids or host['sample_ids']!=ids:
        raise ValueError('Complete configured/launch/manifest/Host partition required')
    closing=completion['worker_batch_closing']
    if completion.get('status')!='rendered_pending_independent_validation' or completion.get('manifest_ref')!=mref or completion.get('training_accepted') is not False or closing.get('status')!='PASS_WORKER_METADATA_CLOSED_MANIFEST_NOT_YET_PUBLISHED' or closing.get('plan_ref')!=plan_ref or closing.get('dependency_map_unchanged') is not True or closing.get('completed_sample_ids')!=sorted(ids):
        raise ValueError('Original worker closing required')
    if host.get('status')!='geometry_verified_visual_acceptance_pending' or host.get('training_accepted') is not False or host.get('manifest_sha256')!=mref['sha256'] or host.get('images')!=len(rows) or host.get('instances')!=sum(len(r['instances']) for r in rows) or host.get('corners')!=8*host['instances']:
        raise ValueError('Bound complete Host geometry receipt required')
    if timing.get('images')!=len(rows) or timing.get('training_accepted') is not False:raise ValueError('Successful launcher final timing mismatch')
    for row,entry in zip(rows,chosen):
        original.check_presence(row,entry)
        if row.get('status')!='render_candidate_not_training_accepted' or row.get('training_accepted') is not False or row.get('dataset_generation_allowed') is not False or row.get('asset_backend')!='modular_composed' or row['pose_controls_proof']['plan_ref']!=plan_ref or row['modular_backend']['plan_sha256']!=plan_ref['sha256']:
            raise ValueError('Candidate identity/status/plan mismatch')
    hpath=str((run/'geometry_validation.json').relative_to(ROOT))
    return {'data_dir':str(data.relative_to(ROOT)),'manifest_ref':mref,'entries':chosen,
            'host_ref':{'path':hpath,'sha256':reader.refs[hpath]}}


def reuse128(reader,config,plan):
    """Read only accepted reports/metadata. Original128 RGB/mask bytes are not reopened."""
    if config['accepted128_ref']!=ACCEPTED128:raise ValueError('Exact reviewed root128 acceptance required')
    accepted=reader.json(ACCEPTED128['path'],ACCEPTED128['sha256'])
    report_ref=config['report128_ref'];report=reader.json(report_ref['path'],report_ref['sha256'])
    ids=[e['sample_id'] for e in plan['entries'][:128]]
    if accepted.get('status')!='ROOT_ACCEPTED_ACTUAL128_FOR_REMAINING1872' or accepted.get('plan_ref')!=config['plan_ref'] or accepted.get('sample_ids')!=ids or accepted.get('numerical_report_ref')!=report_ref:
        raise ValueError('Acceptance does not bind exact original128 report')
    if report.get('status')!='PASS_POINTS_MASKS_DATAFLOW_ROOT_VISUAL_PENDING' or report.get('plan_ref')!=config['plan_ref'] or report.get('count')!=128 or report.get('training_accepted') is not False or [r['sample_id'] for r in report['samples']]!=ids or [r['entry'] for r in report['samples']]!=plan['entries'][:128]:
        raise ValueError('Original128 complete report mismatch')
    parent=bounded(report_ref['path']).parent
    index=reader.json(parent/'MANIFEST_INDEX.json')
    if index!={'plan_ref':config['plan_ref'],'sample_ids':ids,'manifest_refs':report['manifest_refs']}:raise ValueError('Original128 manifest index mismatch')
    for path,sha in report['parent_input_refs'].items():
        if Path(path).suffix.lower() in ('.png','.jpg','.jpeg'):raise ValueError('Unexpected image in parent metadata refs; no original128 image reread allowed')
        reader.data(path,sha)
    children=[];samples=[]
    if len(report['manifest_refs'])!=len(report['host_refs']):raise ValueError('Original128 Host cardinality mismatch')
    for ordinal,(manifest_ref,host_ref) in enumerate(zip(report['manifest_refs'],report['host_refs'])):
        path=parent/('shard_'+str(ordinal).zfill(2))/'REPORT.json';child=reader.json(path)
        if child.get('status')!='PASS_SHARD_POINTS_MASKS_CONSUMER_VISUAL_PENDING' or child.get('manifest_ref')!=manifest_ref or child.get('consumer_source_refs')!=report['consumer_source_refs'] or child.get('training_accepted') is not False:
            raise ValueError('Original128 child report mismatch')
        manifest=reader.data(manifest_ref['path'],manifest_ref['sha256'])
        rows=[json.loads(x) for x in manifest.decode().splitlines() if x.strip()];host=reader.json(host_ref['path'],host_ref['sha256'])
        if [r['sample_id'] for r in rows]!=[r['sample_id'] for r in child['samples']] or host.get('sample_ids')!=[r['sample_id'] for r in rows] or host.get('manifest_sha256')!=manifest_ref['sha256']:
            raise ValueError('Original128 child/manifest/Host identity mismatch')
        for row in rows:
            expected={row['image']:row['rgb_sha256']}
            for instance in row['instances']:
                for key in ('mask_visible','mask_amodal_expanded'):expected[instance[key]]=instance[key+'_sha256']
            if any(child['input_refs'].get(p)!=h for p,h in expected.items()):raise ValueError('Original128 child lacks original image identity proof')
        samples.extend(child['samples']);children.append({'path':str(path.relative_to(ROOT)),'sha256':reader.refs[str(path.relative_to(ROOT))]})
    by_id={r['sample_id']:r for r in samples}
    if len(samples)!=128 or len(by_id)!=128 or [by_id.get(sid) for sid in ids]!=report['samples']:raise ValueError('Original128 child proof partition differs from parent')
    return report,children


def make_sheets(output,results):
    from PIL import Image
    sheets=[]
    for begin in range(0,len(results),16):
        batch=results[begin:begin+16];sheet=Image.new('RGB',(1920,600*((len(batch)+3)//4)),(25,25,25))
        for i,row in enumerate(batch):
            with Image.open(bounded(row['tile'])) as tile:sheet.paste(tile,((i%4)*480,(i//4)*600))
        name='sheet_'+str(len(sheets)+1).zfill(2)+'.jpg';sheet.save(output/name,quality=94);sheets.append(name)
    return sheets



def progress(output,completed,problems,pending,active):
    """Parent-only atomic progress. Never a full validation/acceptance receipt."""
    passed=[i for i,(_,r) in completed.items() if r['status']=='PASS_SHARD_POINTS_MASKS_CONSUMER_VISUAL_PENDING']
    rejected={p['ordinal'] for p in problems if p['status'].startswith('REJECT')}
    not_started={p['ordinal'] for p in problems if p['status']=='NOT_STARTED_INITIAL_GATE_FAILED'}
    errors={p['ordinal'] for p in problems if not p['status'].startswith('REJECT') and p['status']!='NOT_STARTED_INITIAL_GATE_FAILED'}
    ended=set(completed)|rejected|errors|not_started
    children=[{'ordinal':i,'status':r['status'],'report_ref':ref(output/('shard_'+str(i).zfill(2))/'REPORT.json'),
               'passed_samples':len(r.get('samples',[])),'replacement_requests':len(r.get('replacement_requests',[]))}
              for i,(_,r) in sorted(completed.items())]
    payload={'status':'QUALITY_PROGRESS_NOT_COMPLETION_OR_ACCEPTANCE','passed_shards':len(passed),
             'rejected_shards':len(rejected),'error_shards':len(errors),'not_started_shards':len(not_started),'not_started_ordinals':sorted(not_started),'completed_ordinals':sorted(ended),
             'passed_ordinals':sorted(passed),'rejected_ordinals':sorted(rejected),'error_ordinals':sorted(errors),
             'pending_ordinals':sorted(pending),'active_ordinals':sorted(i for i,_ in active.values()),
             'passed_new_samples':sum(len(completed[i][1]['samples']) for i in passed),
             'reported_new_samples':sum(len(r.get('samples',[]))+len(r.get('replacement_requests',[])) for _,r in completed.values()),
             'reused_accepted_samples':128,'expected_new_samples':1872,'expected_total_samples':2000,
             'children':children,'problem_summaries':[{'ordinal':p['ordinal'],'status':p['status'],'reason':p.get('reason','')[:300]} for p in problems],'training_accepted':False}
    target=output/'PROGRESS.json';temp=output/('PROGRESS.'+str(os.getpid())+'.tmp')
    if target.is_symlink() or temp.exists() or temp.is_symlink():raise ValueError('Unexpected progress publication path')
    write_new(temp,payload)
    os.replace(temp,target)


def run(config,out):
    """Wait for and check only configured shards. Never start/retry/stop generation."""
    if not __debug__:raise RuntimeError('Assertions must remain enabled')
    if Path.cwd().resolve()!=ROOT:raise ValueError('Exact project root required')
    config=deepcopy(config);start=time.perf_counter();reader=original.ReadSet()
    plan_ref=config['plan_ref'];plan=reader.json(plan_ref['path'],plan_ref['sha256'])
    if plan['count']!=2000 or len(plan['entries'])!=2000:raise ValueError('Exact original2000 required')
    old,child_refs=reuse128(reader,config,plan)
    model_ref=config.get('config_ref',old['config_ref'])
    if model_ref!=old['config_ref']:raise ValueError('Same original128 consumer configuration required')
    consumers=old['consumer_source_refs'];module_ref=config['stream_module_ref'];quality_ref=config['quality_module_ref']
    verify_code(module_ref,quality_ref,consumers)
    for r in [module_ref,quality_ref,BASE_REF,base.AID_REF,model_ref,*consumers]:reader.data(r['path'],r['sha256'])
    shards=config['shards'];planned={e['sample_id']:e for e in plan['entries']}
    ids=[sid for shard in shards for sid in shard['expected_sample_ids']]
    if len(shards)!=40 or len({str(bounded(s['data_dir'])) for s in shards})!=40 or len(ids)!=1872 or len(set(ids))!=1872 or set(ids)!={e['sample_id'] for e in plan['entries'][128:]}:
        raise ValueError('Exact forty disjoint configured shards covering remaining1872 required')
    initial_gate_shards=config.get('initial_gate_shards',4)
    if type(initial_gate_shards) is not int or initial_gate_shards!=4:raise ValueError('Exactly four initial quality-gate shards required')
    timeout=float(config['timeout_seconds']);poll=float(config.get('poll_seconds',5))
    if not 0<timeout<=604800 or not .1<=poll<=30:raise ValueError('Explicit bounded timeout/poll required')
    output=bounded(out)
    if output==ROOT/'outputs' or not output.is_relative_to(ROOT/'outputs') or output.exists():raise ValueError('Fresh exclusive outputs directory required')
    output.mkdir(parents=True)
    write_new(output/'STARTED.json',{'plan_ref':plan_ref,'accepted128_ref':ACCEPTED128,'report128_ref':config['report128_ref'],
              'config':config,'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
              'workers':4,'threads_each':2,'CPU_budget':8,'gpu_uuid':GPU3_UUID,'training_accepted':False})
    pending={i:shard for i,shard in enumerate(shards)};active={};completed={};problems=[]
    def stop_waiting_after_initial_gate_failure():
        failed=[p['ordinal'] for p in problems if p['ordinal']<initial_gate_shards and p['status']!='NOT_STARTED_INITIAL_GATE_FAILED']
        if not failed or not pending:return
        for ordinal,shard in pending.items():
            problems.append({'status':'NOT_STARTED_INITIAL_GATE_FAILED','ordinal':ordinal,
                             'initial_gate_member':ordinal<initial_gate_shards,'failed_initial_ordinals':sorted(set(failed)),
                             'data_dir':shard['data_dir'],'expected_sample_ids':shard['expected_sample_ids']})
        pending.clear()
        progress(output,completed,problems,pending,active)
    deadline=time.monotonic()+timeout
    progress(output,completed,problems,pending,active)
    try:
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker) as pool:
            while pending or active:
                stop_waiting_after_initial_gate_failure()
                for ordinal,shard in list(pending.items()):
                    if len(active)>=4:break
                    from data.action4_synthesis import _run_output_dir
                    data=bounded(shard['data_dir']);runtime=bounded(_run_output_dir(data))
                    failures=[bounded(runtime/'worker_batch_failure.json'),bounded(runtime/'composition_failure.json')]
                    failure=next((p for p in failures if p.exists()),None)
                    if failure is not None:
                        reader.data(failure);problems.append({'status':'GENERATION_FAILED','ordinal':ordinal,'data_dir':shard['data_dir'],'evidence_ref':ref(failure),'expected_sample_ids':shard['expected_sample_ids']});del pending[ordinal];progress(output,completed,problems,pending,active)
                        if ordinal<initial_gate_shards:break
                        continue
                    required=[bounded(p) for p in (data/'manifest.jsonl',runtime/'fast_launch_timing.json',runtime/'render_completion.json',runtime/'geometry_validation.json',runtime/'launch.json')]
                    if not all(p.is_file() for p in required):continue
                    try:
                        job=snapshot(reader,plan,plan_ref,shard)
                        job.update(ordinal=ordinal,consumer_source_refs=consumers,quality_module_ref=quality_ref)
                        future=pool.submit(check_shard,job,str(output),model_ref,module_ref)
                        active[future]=(ordinal,job);del pending[ordinal]
                    except ValueError as exc:
                        failed_manifest=ref(data/'manifest.jsonl')
                        requests=[rejection(planned[sid],failed_manifest,exc) for sid in shard['expected_sample_ids']]
                        problems.append({'status':'REJECT_SHARD_SNAPSHOT','ordinal':ordinal,'reason':repr(exc),'replacement_requests':requests});del pending[ordinal];progress(output,completed,problems,pending,active)
                        if ordinal<initial_gate_shards:break
                    except Exception as exc:
                        problems.append({'status':'CHECK_ERROR','ordinal':ordinal,'reason':repr(exc)});del pending[ordinal];progress(output,completed,problems,pending,active)
                        if ordinal<initial_gate_shards:break
                stop_waiting_after_initial_gate_failure()
                if active:
                    done,_=wait(active,timeout=min(poll,max(0.,deadline-time.monotonic())) if pending else poll,return_when=FIRST_COMPLETED)
                    for future in done:
                        ordinal,job=active.pop(future)
                        try:
                            report=future.result();completed[ordinal]=(job,report)
                            if report['status']!='PASS_SHARD_POINTS_MASKS_CONSUMER_VISUAL_PENDING':problems.append({'status':report['status'],'ordinal':ordinal,'report_ref':ref(output/('shard_'+str(ordinal).zfill(2))/'REPORT.json')})
                        except Exception as exc:problems.append({'status':'CHECK_ERROR','ordinal':ordinal,'reason':repr(exc)})
                        progress(output,completed,problems,pending,active)
                elif pending:time.sleep(poll)
                stop_waiting_after_initial_gate_failure()
                if pending and time.monotonic()>=deadline:
                    problems.extend({'status':'TIMEOUT_INCOMPLETE','ordinal':i,'data_dir':s['data_dir'],'expected_sample_ids':s['expected_sample_ids']} for i,s in pending.items());pending.clear();progress(output,completed,problems,pending,active)
        verify_code(module_ref,quality_ref,consumers)
        if problems or len(completed)!=40:
            parent_refs=reader.close()
            report={'status':'QUALITY_INCOMPLETE_REPLACEMENT_OR_OPERATION_REVIEW_REQUIRED','plan_ref':plan_ref,'expected_count':2000,
                    'reused128':128,'successful_new_samples':sum(len(r['samples']) for _,r in completed.values() if r['status']=='PASS_SHARD_POINTS_MASKS_CONSUMER_VISUAL_PENDING'),
                    'completed_shards':len(completed),'problems':problems,'parent_input_refs':parent_refs,'training_accepted':False}
            write_new(output/'INCOMPLETE.json',report);return report
        jobs=[completed[i][0] for i in range(40)];reports=[completed[i][1] for i in range(40)]
        results=old['samples']+[r for report in reports for r in report['samples']]
        by_id={r['sample_id']:r for r in results};expected=[e['sample_id'] for e in plan['entries']]
        if len(results)!=2000 or len(by_id)!=2000 or set(by_id)!=set(expected):raise ValueError('Incomplete/duplicate final2000')
        results=[by_id[sid] for sid in expected]
        if [r['entry'] for r in results]!=plan['entries']:raise ValueError('Final frozen entry order differs')
        sheets=make_sheets(output,results)
        manifest_refs=old['manifest_refs']+[j['manifest_ref'] for j in jobs]
        hosts=old['host_refs']+[j['host_ref'] for j in jobs]
        child_refs+= [ref(output/('shard_'+str(i).zfill(2))/'REPORT.json') for i in range(40)]
        for r in child_refs:reader.json(r['path'],r['sha256'])
        for ordinal,report in enumerate(reports):
            path=output/('shard_'+str(ordinal).zfill(2))/'REPORT.json'
            if reader.json(path)!=report:raise ValueError('Written GPU child report differs from returned proof')
        parent_refs=reader.close();verify_code(module_ref,quality_ref,consumers)
        index={'plan_ref':plan_ref,'sample_ids':expected,'manifest_refs':manifest_refs};write_new(output/'MANIFEST_INDEX.json',index)
        report={'status':'PASS_POINTS_MASKS_DATAFLOW_ROOT_VISUAL_PENDING','plan_ref':plan_ref,'count':2000,
                'instance_count':sum(len(r['instances']) for r in results),'samples':results,
                'split_counts':{s:sum(r['entry']['split']==s for r in results) for s in ('train','eval','test')},'sheets':sheets,
                'manifest_refs':manifest_refs,'host_refs':hosts,'parent_input_refs':parent_refs,'module_ref':module_ref,
                'config_ref':model_ref,'consumer_source_refs':consumers,'workers':4,'threads_each':2,
                'elapsed_seconds':time.perf_counter()-start,'max_native_round_trip_error_px':max(r['consumer']['max_native_round_trip_error_px'] for r in results),
                'source_images_labels_modified':False,'root_visual_review_pending':True,'model_predictions_tested':False,'training_accepted':False,
                'accepted128_ref':ACCEPTED128,'reused128_report_ref':config['report128_ref'],'shard_report_refs':child_refs,
                'quality_source_refs':[module_ref,quality_ref,BASE_REF],
                'execution_coverage':{'original128':'reused accepted CPU report; RGB/masks not reopened','remaining1872':'fresh GPU3 FP64 plus unchanged CPU mask/consumer checks'},
                'gpu_compute_receipts':[r['gpu_compute'] for r in reports],
                'limitations':['Numerical checks do not replace root visual review or establish model accuracy.','Accepted128 bytes are evidence-reused, not freshly rehashed; final assembly must check their image identities.','No automatic acceptance or replacement is performed.']}
        if report['split_counts']!=plan['split_counts']:raise ValueError('Final split coverage differs')
        write_new(output/'REPORT.json',report);return {k:report[k] for k in ('status','count','instance_count','elapsed_seconds')}
    except BaseException as exc:
        write_new(output/'FAILED.json',{'status':'CHECK_ERROR','error':repr(exc),'traceback':traceback.format_exc(),'training_accepted':False});raise


def check_shard(job,out,config_ref,module_ref):
 from PIL import Image,ImageDraw
 import numpy as np
 from data.action4_dataset import Action4Dataset
 start=time.perf_counter();worker_out=bounded(out)/('shard_'+str(job['ordinal']).zfill(2));worker_out.mkdir();phase='source'
 try:
  verify_code(module_ref,job['quality_module_ref'],job['consumer_source_refs'])
  aid=gpu_aid(job['quality_module_ref']);reader=aid.ReadSet()
  for r in [module_ref,BASE_REF,config_ref,*job['consumer_source_refs']]:reader.data(r['path'],r['sha256'])
  phase='dataset'
  for path,h in aid.DEPENDENCIES.items():reader.data(path,h)
  raw=reader.data(job['manifest_ref']['path'],job['manifest_ref']['sha256']);rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()];config=readref(config_ref)
  ds=Action4Dataset(ROOT,job['manifest_ref']['path'],allow_diagnostic=True)
  if ds.records!=rows:raise ValueError('Dataset loaded different records')
  try:Action4Dataset(ROOT,job['manifest_ref']['path'])
  except ValueError as e:
   if 'unaccepted' not in str(e):raise
  else:raise AssertionError('Formal training admitted unaccepted candidates')
  if len(rows)!=len(job['entries']):raise ValueError('Exact record/entry cardinality required')
  results=[];rejects=[];phase='records'
  for n,(row,entry) in enumerate(zip(rows,job['entries'])):
   try:
    aid.check_presence(row,entry);overlay,instances,semantics=aid.validate_record(reader,row,bounded(job['data_dir']));item=ds.collate_fn([ds[n]])[0];metrics=consumer_check(item,config)
    sid=row['sample_id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+',sid):raise ValueError('Unsafe sample filename')
    overlay_path=worker_out/(sid+'_overlay.png');overlay.save(overlay_path)
    tile=Image.new('RGB',(480,600),(25,25,25));thumb=overlay.copy();thumb.thumbnail((480,365));tile.paste(thumb,((480-thumb.width)//2,0));draw=ImageDraw.Draw(tile);draw.text((4,367),sid+' '+entry['scene_family'],fill='white')
    for j,instance in enumerate(instances):
     uv=np.asarray(instance['independent_points_native_px']);low=uv.min(0);high=uv.max(0);box=(max(0,int(np.floor(low[0]))-32),max(0,int(np.floor(low[1]))-32),min(overlay.width,int(np.ceil(high[0]))+33),min(overlay.height,int(np.ceil(high[1]))+33));assert box[2]>box[0] and box[3]>box[1]
     crop=overlay.crop(box);crop.thumbnail((238,195));tile.paste(crop,(j*240+(238-crop.width)//2,390));draw.text((j*240+4,584),'physical ID '+str(instance['instance_id']),fill='white')
    tile_path=worker_out/(sid+'_tile.png');tile.save(tile_path)
    results.append({'sample_id':sid,'entry':entry,'instances':instances,'mask_semantics':semantics,'consumer':metrics,'overlay':str(overlay_path.relative_to(ROOT)),'tile':str(tile_path.relative_to(ROOT))})
   except (ValueError,AssertionError) as exc:
    rejects.append(rejection(entry,job['manifest_ref'],repr(exc)))
    continue
  phase='closing';identities=reader.close()
  verify_code(module_ref,job['quality_module_ref'],job['consumer_source_refs'])
  if ref(config_ref['path'])!=config_ref:raise ValueError('Consumer config changed')
  report={'status':'REJECT_SHARD_REVIEW_REQUIRED' if rejects else 'PASS_SHARD_POINTS_MASKS_CONSUMER_VISUAL_PENDING','manifest_ref':job['manifest_ref'],'samples':results,'input_refs':identities,'consumer_source_refs':job['consumer_source_refs'],'elapsed_seconds':time.perf_counter()-start,'training_accepted':False,'replacement_requests':rejects,'gpu_compute':aid.receipt(),'quality_source_refs':[module_ref,job['quality_module_ref'],BASE_REF]};write_new(worker_out/'REPORT.json',report);return report
 except Exception as e:
  status='REJECT_SHARD_REVIEW_REQUIRED' if isinstance(e,ValueError) and phase=='dataset' else 'CHECK_ERROR'
  report={'status':status,'manifest_ref':job['manifest_ref'],'samples':[],'error':repr(e),'phase':phase,'traceback':traceback.format_exc(),'training_accepted':False}
  if status=='REJECT_SHARD_REVIEW_REQUIRED':report['replacement_requests']=[rejection(entry,job['manifest_ref'],'Shard-level data failure; individual cause needs review: '+repr(e)) for entry in job['entries']]
  write_new(worker_out/'FAILED.json',report);write_new(worker_out/'REPORT.json',report);return report
