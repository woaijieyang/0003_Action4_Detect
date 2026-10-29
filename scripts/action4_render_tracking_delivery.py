"""Render complete offline B0/SAM tracking beside the supplied demo.

Consumes existing predictions; performs no model inference or A0 generation.
Frame-index alignment is explicit, including the four-frame bootstrap backfill.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from PIL import Image, ImageDraw, ImageFont

R=Path('/data/zy/vln/a-test')
GPU='GPU-8c93aadc-2a0a-665c-46c3-dbd3f063e0d8'

def inside(value):
    p=Path(value);p=p if p.is_absolute() else R/p
    if p.is_symlink() or not p.resolve().is_relative_to(R):raise ValueError('Outside project')
    return p

def sha(path):
    with inside(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def read(path):return json.loads(inside(path).read_text())

def source_ref(path):return {'path':str(inside(path).relative_to(R)),'sha256':sha(path)}

def write(path,value):inside(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def bootstrap_rows(rows,native):
    """Offline backfill uses actual current-frame B0 candidates, never held boxes.

    Confirmation occurs at frame4. This is disclosed separately from causal SAM.
    """
    cfg=read(native/'source/config.json')
    seed_ref=cfg['instance_seed_sources']['track_00']['seeds']
    if sha(seed_ref['path'])!=seed_ref['sha256']:raise ValueError('Seed evidence changed')
    seeds=read(seed_ref['path'])['seeds']
    done=read(cfg['b0_evidence']['path'])
    if sha(cfg['b0_evidence']['path'])!=cfg['b0_evidence']['sha256']:raise ValueError('B0 evidence changed')
    pool_ref=done['pools']
    if sha(pool_ref['path'])!=pool_ref['sha256']:raise ValueError('B0 pool changed')
    proofs=[]
    with inside(pool_ref['path']).open() as source:
        for fid in range(4):
            pool=json.loads(source.readline())
            if pool['frame']!=rows[fid]['frame'] or rows[fid]['fused']['tracks']:raise ValueError('Bootstrap domain differs')
            tracks=[];used=set()
            for seed in seeds:
                evidence=[]
                for token in seed['evidence_ids']:
                    match=re.fullmatch(r'(?:b0:)?frame(\d+):grid:(\d+)',token)
                    if match and int(match[1])==fid:evidence.append(int(match[2]))
                if len(evidence)!=1:raise ValueError('One recorded confirmation candidate per frame required')
                grid=evidence[0]
                candidates=[x for x in pool['pool']['association_candidates'] if x['grid_index']==grid]
                if len(candidates)!=1 or grid in used:raise ValueError('Duplicate/missing bootstrap evidence')
                used.add(grid);raw=dict(candidates[0]);cid=f'b0:frame{fid}:grid:{grid}';raw['candidate_id']=cid
                if raw['score']<.9 or raw['refined_quality_score']<.8:raise ValueError('Recorded strong bootstrap evidence differs')
                tracks.append({'track_id':seed['track_id'],'status':'observed','bbox_for_roi':raw['bbox_full_xyxy'],
                    'candidate_id':cid,'raw_detection':raw,'source':'offline_B0_confirmation_chain',
                    'sam_current_support':False,'confirmation_frame':4,'offline_future_confirmation':True,
                    'A0_connected':False,'full_box_accuracy_verified':False})
            rows[fid]=dict(rows[fid],status='offline_bootstrap_backfill',
                fused={'frame_id':fid,'mode':'video_inference','tracks':tracks})
            proofs.append({'frame':fid,'confirmation_frame':4,'candidate_ids':[x['candidate_id'] for x in tracks]})
    return proofs

def command_text(rows):
    result=[]
    for fid,row in enumerate(rows):
        if row['frame']['global_frame_id']!=fid or row['frame']['pts']!=fid*400:raise ValueError('Frame/PTS mismatch')
        tracks={x['track_id']:x for x in row['fused']['tracks']}
        commands=[]
        for index,tid in enumerate(('track_00','track_01')):
            box=tracks.get(tid,{}).get('bbox_for_roi')
            if box is None:x,y,w,h=-9999,-9999,1,1
            else:
                if len(box)!=4 or not all(math.isfinite(v) for v in box) or not(box[2]>box[0] and box[3]>box[1]):raise ValueError('Invalid display box')
                x,y=box[:2];w,h=box[2]-x,box[3]-y
            commands.extend(f'drawbox@cam{index} {k} {v:.6f}' for k,v in [('x',x),('y',y),('w',w),('h',h)])
        # A sub-nanosecond lead avoids decimal rounding putting a command AFTER
        # its exact frame PTS. The prior frame is ~33ms earlier.
        timestamp=max(0,fid*400/11988-1e-9)
        result.append(f'{timestamp:.12f} '+', '.join(commands)+';')
    return '\n'.join(result)+'\n'

def resolve_bootstrap_source(replay,complete,audit):
    """A reanchor run uses its own predictions and the original seed evidence."""
    if complete.get("status")!="FULL2850_REANCHOR_CANDIDATE_COMPLETE_NOT_ACCURACY":
        return inside(audit["native_job"])
    manifest=read(replay/"source/SNAPSHOT_MANIFEST.json")
    cleanup=read(replay/"run/CLEANUP.json")
    if (complete.get("closed") is not True or cleanup.get("failed") is not False
            or complete.get("source_identity")!=manifest.get("identity")
            or not isinstance(complete.get("events"),list)
            or any(e.get("outcome")!="native_bound" for e in complete["events"])):
        raise ValueError("Incomplete or failed generation run cannot be rendered as complete")
    cfg=read(replay/"source/config.json")
    baseline=inside(cfg["reanchor"]["baseline_job"])
    if read(baseline/"source/config.json")["b0"]!=cfg["b0"]:
        raise ValueError("Bootstrap and current frozen B0 identities differ")
    ref=cfg["reanchor"].get("baseline_manifest")
    if ref is not None and (inside(ref["path"])!=baseline/"source/SNAPSHOT_MANIFEST.json" or sha(ref["path"])!=ref["sha256"]):
        raise ValueError("Original native bootstrap snapshot differs")
    return baseline

def run(replay_root,output):
    if Path.cwd().resolve()!=R:raise ValueError('Verified project cwd required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('Physical GPU5 UUID required')
    replay=inside(replay_root);complete=read(replay/'run/COMPLETE.json')
    if complete['frames']!=2850 or (replay/'run/FAILED.json').exists():raise ValueError('Complete replay required')
    audit=read(replay/'run/AUDIT.json');native=resolve_bootstrap_source(replay,complete,audit)
    if audit['lost_spans']:raise ValueError('Lost intervals require review before delivery')
    predictions=replay/'run/FUSED_BOXES.jsonl';rows=[json.loads(x) for x in predictions.read_text().splitlines()]
    if len(rows)!=2850:raise ValueError('Full2850 required')
    out=inside(output);out.mkdir(parents=True,exist_ok=False)
    proofs=bootstrap_rows(rows,native)
    delivery=out/'delivery_predictions.jsonl'
    with delivery.open('x') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    commands=out/'boxes.sendcmd';commands.write_text(command_text(rows))
    manifest=read('data/action4_real/cache_v001/frames_manifest.json');source=inside(manifest['source']['source_video'])
    demo=inside('data/action4_reference/demo_standard.mp4')
    if manifest['source']['frame_count']!=2850 or manifest['source']['observed_cfr_fps']!='2997/100':raise ValueError('Raw video domain differs')
    ff=R/'runtime/ffmpeg/bin/ffmpeg';fp=R/'runtime/ffmpeg/bin/ffprobe'
    probes={}
    for name,path in [('raw',source),('demo',demo)]:
        probes[name]=json.loads(subprocess.check_output([str(fp),'-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,avg_frame_rate','-of','json',str(path)],text=True))
        if int(probes[name]['streams'][0]['nb_frames'])!=2850:raise ValueError('Input frame count differs')
    if (probes['raw']['streams'][0]['width'],probes['raw']['streams'][0]['height'])!=(3248,2464):raise ValueError('Native drawing space differs')
    header=Image.new('RGB',(1900,30),'black');draw=ImageDraw.Draw(header);font=ImageFont.load_default(size=18)
    draw.text((8,5),'Demo reference',fill='white',font=font)
    draw.text((958,5),'B0 + SAM3.1 tracking candidate | ID0 green / ID1 orange | no A0',fill='white',font=font)
    header.save(out/'header.png')
    filters=(f'[0:v]settb=1/11988,setpts=N*400,scale=950:720,setsar=1[left];'
        f'[1:v]settb=1/11988,setpts=N*400,sendcmd=f={commands},'
        'drawbox@cam0=x=-9999:y=-9999:w=1:h=1:c=0x00FF00:t=8,'
        'drawbox@cam1=x=-9999:y=-9999:w=1:h=1:c=0xFF9500:t=8,'
        'scale=950:720,setsar=1[right];'
        '[left][right]hstack=inputs=2:shortest=1,pad=1900:750:0:30:color=black[base];'
        '[base][2:v]overlay=0:0:shortest=1[out]')
    cmd=[str(ff),'-hide_banner','-loglevel','warning','-nostdin','-n','-threads','2','-i',str(demo),
        '-threads','2','-i',str(source),'-loop','1','-framerate','2997/100','-i',str(out/'header.png'),
        '-filter_complex_threads','4','-filter_complex',filters,'-map','[out]','-an','-frames:v','2850',
        '-r','2997/100','-fps_mode','cfr','-c:v','h264_nvenc','-gpu','0','-preset','p4','-cq','19',
        '-pix_fmt','yuv420p','-video_track_timescale','11988','-progress',str(out/'progress.log'),str(out/'comparison.mp4')]
    started=time.time();write(out/'RUN.json',{'command':cmd,'GPU_physical':5,'GPU_logical':0,'GPU_UUID':GPU,
        'prediction_source':source_ref(predictions),'audit_source':source_ref(replay/'run/AUDIT.json'),
        'frame_alignment':'frame index i->PTS400*i in1/11988 for BOTH videos; no frame interpolation',
        'offline_bootstrap':proofs,'reanchor_events':[{'event_id':e['event_id'],'seed_frame':e['seed_frame'],'trigger_frame':e['trigger_frame'],'generation':e['proposed_generation']} for e in complete.get('events',[])],'model_calls':0,'A0_included':False,'input_probes':probes})
    with (out/'encode.log').open('x') as log:
        proc=subprocess.run(cmd,cwd=R,stdout=log,stderr=log)
    if proc.returncode:
        write(out/'FAILED.json',{'stage':'encoding','returncode':proc.returncode});raise RuntimeError('Encoding failed')
    encoded=json.loads(subprocess.check_output([str(fp),'-v','error','-threads','2','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames,avg_frame_rate,width,height,time_base','-of','json',str(out/'comparison.mp4')],text=True))
    stream=encoded['streams'][0]
    if int(stream['nb_read_frames'])!=2850 or stream['avg_frame_rate']!='2997/100':raise ValueError('Encoded frame domain differs')
    samples=[0,4,116,118,1195,1289,1374,2400,2849]
    select='select='+ '+'.join(f'eq(n\\,{fid})' for fid in samples)
    subprocess.run([str(ff),'-v','error','-threads','2','-i',str(out/'comparison.mp4'),'-vf',select,
        '-fps_mode','passthrough','-threads','1','-start_number','0',str(out/'sample_%02d.png')],check=True)
    for index,fid in enumerate(samples):
        (out/f'sample_{index:02d}.png').rename(out/f'frame_{fid:08d}.png')
    supported=sum(len(row['fused']['tracks'])==2 and all(t['status'] in ('observed','sam_supported')
        for t in row['fused']['tracks']) for row in rows[4:])
    result={'status':'FULL2850_TRACKING_VIDEO_DELIVERED_PENDING_VISUAL_ACCURACY_REVIEW','frames':2850,
        'raw_replay_source':str(replay.relative_to(R)),'geometry_mode':audit.get('geometry_mode'),
        'two_current_support_frames_after_seed':supported,'offline_bootstrap_frames':4,'causal_from_frame_zero':False,
        'counts':audit['counts'],'lost_spans':audit['lost_spans'],'reanchor_event_count':len(complete.get('events',[])),'model_calls':0,'A0_included':False,
        'demo_is_reference_not_certified_GT':True,'encoded':encoded,'elapsed_seconds':time.time()-started}
    write(out/'COMPLETE.json',result);print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay-root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.replay_root,args.output)
