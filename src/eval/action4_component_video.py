"""Streaming CFR comparison video for fixed independent components; diagnostic."""
from pathlib import Path
import json,subprocess,time
from fractions import Fraction
import numpy as np
import torch
from PIL import Image,ImageDraw
from data.action4_paths import PROJECT_ROOT as R,contained_path
from eval.action4_geometry_pipeline import GeometryPipeline
from eval.action4_video import draw_cuboids
from model.training.action4_runtime import atomic_json


def run(output,limit=None):
    torch.set_num_threads(2);out=contained_path(R,R/output,exists=True)
    manifest=json.loads((R/'data/action4_real/cache_v001/frames_manifest.json').read_text());frames=manifest['frames'];source=manifest['source']
    assert len(frames)==2850 and manifest['complete']
    assert all(f['frame_id']==i and f['pts']==i*400 and f['duration_pts']==400 for i,f in enumerate(frames))
    assert Fraction(source['time_base'])*400==1/Fraction(source['observed_cfr_fps'])
    if limit is not None:frames=frames[:limit]
    checkpoints=R/'outputs/action4_current_geometry_comparison_v001'
    pipeline=GeometryPipeline(checkpoints/'B0.pt',checkpoints/'A0.pt',checkpoints/'residual.pt','cuda:0')
    context=json.loads((R/'outputs/action4_execution_source_release_v006/source_context.json').read_text());K=np.array(context['K_native'])
    record=json.loads((R/'data/eval/annotations.jsonl').read_text().splitlines()[0]);canonical=np.array(record['corners_3d_mm'])/1000
    fps=source['observed_cfr_fps'];video=out/'comparison.mp4';partial=out/'comparison.partial.mp4'
    command=[str(R/'runtime/ffmpeg/bin/ffmpeg'),'-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1900x750','-r',fps,'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-video_track_timescale','11988','-movie_timescale','11988','-movflags','+faststart',str(partial)]
    begin=time.monotonic()
    with (out/'encode.log').open('x') as log,(out/'predictions.jsonl').open('x') as predictions:
        encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            for i,frame in enumerate(frames):
                image=contained_path(R,R/frame['image'],exists=True)
                with Image.open(image) as im:rgb=im.convert('RGB')
                assert rgb.size==(3248,2464)
                variants=pipeline.predict(torch.from_numpy(np.array(rgb)).permute(2,0,1),K,canonical)
                predictions.write(json.dumps({'frame_id':frame['frame_id'],'pts':frame['pts'],'time_base':source['time_base'],'variants':variants},allow_nan=False)+'\n')
                panel=Image.new('RGB',(1900,750))
                for column,name in enumerate(['raw','visibility_residual']):
                    rendered=draw_cuboids(rgb,variants[name],line_width=6).resize((950,720));panel.paste(rendered,(column*950,30))
                    ImageDraw.Draw(panel).text((column*950+8,8),f'{name} | frame {frame["frame_id"]} | DIAGNOSTIC, assumed K, no temporal smoothing',fill='white')
                encoder.stdin.write(panel.tobytes())
                if i==0 or (i+1)%25==0 or i+1==len(frames):
                    predictions.flush();atomic_json(out/'STATUS.json',{'completed_frames':i+1,'total':len(frames),'elapsed_seconds':time.monotonic()-begin})
            encoder.stdin.close()
            if encoder.wait()!=0:raise RuntimeError('Encoder failed; inspect encode.log')
        except BaseException:
            if encoder.stdin and not encoder.stdin.closed:encoder.stdin.close()
            if encoder.poll() is None:encoder.terminate()
            encoder.wait();raise
    probe=json.loads(subprocess.check_output([str(R/'runtime/ffmpeg/bin/ffprobe'),'-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames,avg_frame_rate,time_base,duration_ts,width,height','-of','json',str(partial)],text=True))['streams'][0]
    assert int(probe['nb_read_frames'])==len(frames)
    assert Fraction(probe['avg_frame_rate'])==Fraction(fps)
    assert Fraction(probe['time_base'])*int(probe['duration_ts'])==len(frames)/Fraction(fps)
    partial.rename(video)
    atomic_json(out/'COMPLETE.json',{'status':'VIDEO_IO_COMPLETE_NOT_ACCURACY_ACCEPTED','frames':len(frames),'full_video':len(frames)==2850,'probe':probe,'source_checkpoints':str(checkpoints.relative_to(R)),'intrinsics':'assumed synthetic source K, not measured real calibration','frame_ids_and_pts_checked':True,'smoothing':None,'fit_missing_fallback':pipeline.fit_missing,'nonfinite_fallback':pipeline.nonfinite_fallback})

if __name__=='__main__':
    import argparse,traceback
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--limit',type=int);a=p.parse_args()
    try:run(a.output,a.limit)
    except BaseException:
        atomic_json(R/a.output/'FAILURE.json',{'traceback':traceback.format_exc()});raise
