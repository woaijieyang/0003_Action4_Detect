"""Saved direct corner-head coordinates + fixed-ID 2D tracking, never PnP.

Extraction and temporal filtering happen before rendering. Renderer only reads
saved native points. Current B0/SAM ROI and enhanced-head inputs are preserved.
"""
from pathlib import Path
from fractions import Fraction
import copy,json,hashlib,os,subprocess,time
import numpy as np
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.action4_direct_corner_tracking import direct_native_observations,DirectCornerTracker

def bound(p):
    p=contained_path(R,p,exists=True)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p.relative_to(R)),sha256=h)

def read(ref):
    p=contained_path(R,ref['path'],exists=True);data=p.read_bytes()
    if hashlib.sha256(data).hexdigest()!=ref['sha256']:raise ValueError('Bound source changed')
    return json.loads(data)

def extract(cfg):
    out=contained_path(R,cfg['output'],exists=False);out.mkdir(exist_ok=False)
    manifest=read(cfg['frames_manifest']);receipt=read(cfg['corner_trial_receipt']);pair=read(cfg['model_pair'])
    if receipt['frames']!=2850 or len(manifest['frames'])!=2850:raise ValueError('Full2850 source required')
    tracker=DirectCornerTracker(**cfg['filter']);digest=hashlib.sha256();instances=0;points=0;outside=0;outside_roi=0;missing=0;model_identity=None
    start=time.monotonic()
    with contained_path(R,cfg['predictions']['path'],exists=True).open('rb') as src,(out/'predictions.jsonl').open('x') as dst:
        count=0
        for fid,line in enumerate(src):
            digest.update(line);row=json.loads(line);f=manifest['frames'][fid]
            if row['frame_id']!=fid or row['pts']!=f['pts'] or row['RGB_sha256']!=f['image_sha256'] or row['roi_expansion']!=cfg['roi_expansion']:raise ValueError('Frame/PTS/RGB/ROI mismatch')
            pred=row['prediction'];identity=pred['identity']
            if model_identity is None:model_identity=identity
            elif identity!=model_identity:raise ValueError('Model identity changed in source')
            for role in ['A0','B0']:
                if identity['models'][role]['sha256']!=pair['models'][role]['sha256']:raise ValueError('Wrong selected model weights')
            specs={x['prediction_id']:x for x in pred['a0_roi_specs']};observations={x['prediction_id']:x for x in pred['corner_observations']}
            if len(specs)!=len(pred['a0_roi_specs']) or len(observations)!=len(pred['corner_observations']):raise ValueError('Duplicate direct ROI/head records')
            if len({i['track_id'] for i in pred['instances']})!=len(pred['instances']):raise ValueError('Duplicate body track IDs')
            output=dict(frame_id=fid,pts=row['pts'],RGB_sha256=row['RGB_sha256'],time_base=manifest['source']['time_base'],roi_expansion=cfg['roi_expansion'],instances=[])
            for i in pred['instances']:
                pid=i['prediction_id'];item={k:copy.deepcopy(i[k]) for k in ['prediction_id','track_id','frame_id','pts','bbox_for_a0','bbox_b0_raw','bbox_sam_visible','roi_source','roi_decision_id','current_support_status']};instances+=1
                if pid not in specs or pid not in observations:
                    missing+=1;item.update(status='missing_direct_head_observation_no_pose_fallback',corner_ids=list(range(8)),corners_detector_native=None,corners_tracked_native=None);tracker.states.pop(i['track_id'],None)
                else:
                    direct=direct_native_observations(i,observations[pid],specs[pid])
                    if direct['uv224']!=i['corner_observations']['uv224']:raise ValueError('Duplicated saved direct head values differ')
                    t=float(Fraction(row['pts'])*Fraction(manifest['source']['time_base']))
                    filtered=tracker.update(track_id=i['track_id'],frame_id=fid,time_seconds=t,points_native=direct['corners_detector_native'],full_box=i['bbox_for_a0'],corner_ids=direct['corner_ids'])
                    item.update(direct);item.update(filtered);points+=8;uv=np.array(direct['corners_detector_native']);size=np.array([manifest['source']['width'],manifest['source']['height']]);outside+=int(((uv<0)|(uv>=size)).any(1).sum());q=np.array(direct['uv224']);outside_roi+=int(((q<0)|(q>=224)).any(1).sum())
                output['instances'].append(item)
            dst.write(json.dumps(output,allow_nan=False,separators=(',',':'))+'\n');count+=1
    if count!=2850 or digest.hexdigest()!=cfg['predictions']['sha256']:raise ValueError('Incomplete or changed direct-head source')
    result=dict(schema='action4.direct_corner_video.v1',status='FULL2850_DIRECT_HEAD_AND_2D_TRACKING_SAVED',frames=count,instances=instances,direct_points=points,missing_head_instances=missing,detector_points_outside_image=outside,detector_points_outside224=outside_roi,model_identity=model_identity,source=cfg['predictions'],corner_trial_receipt=cfg['corner_trial_receipt'],roi_expansion=cfg['roi_expansion'],filter=cfg['filter'],head_decoder='saved 7x7 peak-local soft argmax in224 pixel centres',PnP_calls=0,pose_projection_calls=0,new_model_calls=0,coordinates_clamped=False,quality_threshold_applied=False,renderer_changes_points=False,predictions=bound(out/'predictions.jsonl'),seconds=time.monotonic()-start,accuracy_accepted=False)
    (out/'COMPLETE.json').write_text(json.dumps(result,indent=2));return result

def render(cfg,receipt):
    from PIL import Image,ImageDraw,ImageFont
    from data.action4_ordered_image_prefetch import iter_rgb_previews
    out=contained_path(R,cfg['video_output'],exists=False);out.mkdir(exist_ok=False)
    manifest=read(cfg['frames_manifest']);source=manifest['source'];fps=source['avg_frame_rate'];tb=Fraction(source['time_base']);step=1/Fraction(fps)/tb
    if step.denominator!=1:raise ValueError('Exact PTS frame interval required')
    if bound(receipt['predictions']['path'])!=receipt['predictions']:raise ValueError('Saved display coordinates changed')
    rows=[json.loads(l) for l in contained_path(R,receipt['predictions']['path'],exists=True).open()]
    if len(rows)!=2850:raise ValueError('Complete display domain required')
    video=out/'comparison.mp4';ffmpeg=R/'runtime/ffmpeg/bin/ffmpeg';ffprobe=R/'runtime/ffmpeg/bin/ffprobe'
    cmd=[str(ffmpeg),'-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1900x750','-r',fps,'-i','pipe:0','-an','-c:v','libx264','-threads',str(cfg['encoder_threads']),'-preset','fast','-crf','18','-pix_fmt','yuv420p','-video_track_timescale','11988','-movie_timescale','11988',str(video)]
    font=ImageFont.load_default(size=14);scale=np.array([950/source['width'],720/source['height']]);edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    log=(out/'encode.log').open('x');proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log);rendered=0
    try:
        previews=iter_rgb_previews(manifest['frames'],image_size=(source['width'],source['height']),workers=cfg['prefetch_workers'],verify_sha=True)
        for row,(f,base) in zip(rows,previews):
            if row['frame_id']!=f['frame_id'] or row['pts']!=f['pts'] or row['RGB_sha256']!=f['image_sha256']:raise ValueError('Renderer source identity mismatch')
            canvas=Image.new('RGB',(1900,750),'black')
            for side,key in enumerate(['corners_detector_native','corners_tracked_native']):
                im=base.copy();draw=ImageDraw.Draw(im)
                for instance in row['instances']:
                    box=np.array(instance['bbox_for_a0'])*np.tile(scale,2);draw.rectangle(box.tolist(),outline='lime',width=2)
                    draw.text((box[0],max(0,box[1]-16)),instance['track_id'],fill='lime',font=font)
                    if instance[key] is None:continue
                    uv=np.asarray(instance[key])*scale;color='#ff5555' if instance['track_id']=='track_00' else '#ffd84d'
                    for a,b in edges:draw.line([tuple(uv[a]),tuple(uv[b])],fill=color,width=1)
                    for cid,(x,y) in enumerate(uv):
                        draw.ellipse((x-3,y-3,x+3,y+3),fill=color,outline='black',width=1);draw.text((x+4,y-8),str(cid),fill=color,stroke_width=1,stroke_fill='black',font=font)
                canvas.paste(im,(950*side,30));label='DIRECT CORNER HEAD / NO PnP' if side==0 else 'DIRECT HEAD + ID TRACK + 2D SMOOTH / NO PnP'
                ImageDraw.Draw(canvas).text((950*side+6,7),label+' | ROI'+str(cfg['roi_expansion'])+' | frame '+str(row['frame_id']),fill='white',font=font)
            proc.stdin.write(canvas.tobytes());rendered+=1
            if rendered%100==0:(out/'PROGRESS.json').write_text(json.dumps(dict(rendered=rendered,expected=2850)))
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError('Encoder failed')
    except BaseException:
        if proc.poll() is None:proc.stdin.close();proc.wait()
        (out/'FAILED.json').write_text(json.dumps(dict(rendered=rendered,expected=2850)));raise
    finally:log.close()
    probe=json.loads(subprocess.check_output([str(ffprobe),'-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,avg_frame_rate,time_base,duration_ts','-of','json',str(video)],text=True))['streams'][0]
    if int(probe['nb_frames'])!=2850 or Fraction(probe['avg_frame_rate'])!=Fraction(fps) or Fraction(probe['time_base'])*int(probe['duration_ts'])!=2850/Fraction(fps):raise ValueError('Encoded frame/time mismatch')
    result=dict(status='FULL_DIRECT_HEAD_VIDEO_RENDERED_WITH_TRACK_AND_SMOOTH_NO_PNP',video=bound(video),probe=probe,source=receipt['predictions'],PnP_calls=0,new_model_calls=0,pose_projection_calls=0,accuracy_accepted=False)
    (out/'COMPLETE.json').write_text(json.dumps(result,indent=2));return result
