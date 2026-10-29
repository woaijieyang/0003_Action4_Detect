"""Prepare explicitly registered native tracking clips without rerunning models."""
import argparse,json
from eval.action4_native_clip_prepare import prepare_clips
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True)
    p.add_argument("--clip",nargs=3,action="append",required=True,metavar=("NAME","START","END"))
    p.add_argument("--input-lock");p.add_argument("--predictions");p.add_argument("--max-clip-frames",type=int,choices=[120,360],default=120)
    a=p.parse_args();clips={name:(int(start),int(end)) for name,start,end in a.clip}
    if len(clips)!=len(a.clip):p.error("duplicate clip names")
    print(json.dumps(prepare_clips(a.output,clips,input_lock=a.input_lock,predictions=a.predictions,max_clip_frames=a.max_clip_frames)))
