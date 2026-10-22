"""Independent raw-frame inference shard; never claims a complete video."""
import sys,json,argparse,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
from model.training.action4_runtime import input_path,load_checkpoint,sha256_file,canonical_hash
from eval.action4_video import validate_video_qualification,draw_cuboids
from eval.action4_runner import build_model,predict_images,inference_contract

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--requests',required=True);p.add_argument('--output',required=True);a=p.parse_args();job=json.loads(input_path(R,a.config).read_text());cfg=json.loads(input_path(R,job['video_config']).read_text());ck=input_path(R,job['checkpoint']);manifest_path=input_path(R,job['frame_manifest']);manifest=json.loads(manifest_path.read_text());ids=json.loads(input_path(R,a.requests).read_text());assert ids and len(ids)==len(set(ids)) and all(type(i) is int and 0<=i<len(manifest['frames']) for i in ids);assert manifest['complete'] and [f['frame_id'] for f in manifest['frames']]==list(range(manifest['source']['frame_count']))
 state=load_checkpoint(ck);component,annotation_stage=validate_video_qualification(state,cfg);assert not component;out=R/a.output;assert out.resolve()==out and out.is_relative_to(R/'outputs');out.mkdir(exist_ok=False);drawn=out/'drawn_frames';drawn.mkdir();torch.set_num_threads(2);device='cuda:0';model=build_model(cfg,checkpoint=ck,device=device);mh=sha256_file(ck);ch=canonical_hash(inference_contract(cfg));start=time.monotonic();timings={'read_check_decode_seconds':0.,'inference_seconds':0.,'draw_save_seconds':0.};pixel_refs=[]
 with (out/'predictions_raw.jsonl').open('x') as f:
  for i in ids:
   t=time.monotonic();frame=manifest['frames'][i];path=input_path(R,frame['image']);ih=sha256_file(path);assert ih==frame['image_sha256']
   with Image.open(path) as im:im.verify()
   with Image.open(path) as im:im.load();rgb=im.convert('RGB');assert rgb.size==(manifest['source']['width'],manifest['source']['height']);tensor=torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1).contiguous()
   timings['read_check_decode_seconds']+=time.monotonic()-t;torch.cuda.synchronize();t=time.monotonic();instances=predict_images(model,[tensor],cfg,device)[0];torch.cuda.synchronize();timings['inference_seconds']+=time.monotonic()-t
   record={'frame_id':i,'pts':frame['pts'],'duration_pts':frame['duration_pts'],'time_base':manifest['source']['time_base'],'source_image':frame['image'],'source_image_sha256':ih,'checkpoint_sha256':mh,'inference_config_sha256':ch,'instances':instances,'claim_scope':'model_raw_predictions_pending_visual_review'}
   t=time.monotonic();rendered=draw_cuboids(rgb,instances,line_width=cfg.get('video',{}).get('line_width',6),debug_numbers=False,diagnostic_watermark=False);dest=drawn/f'frame_{i:08d}.png';rendered.save(dest);pixel_refs.append({'frame_id':i,'image':str(dest.relative_to(R)),'sha256':sha256_file(dest)});timings['draw_save_seconds']+=time.monotonic()-t;f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
 report={'status':'FRAME_SHARD_COMPLETE_NOT_FULL_VIDEO','frame_ids':ids,'count':len(ids),'checkpoint_sha256':mh,'inference_config_sha256':ch,'frame_manifest_sha256':sha256_file(manifest_path),'predictions_sha256':sha256_file(out/'predictions_raw.jsonl'),'drawn_frames':pixel_refs,'timings':timings,'elapsed_seconds':time.monotonic()-start,'annotation_stage':annotation_stage,'smoothing':None,'association':None,'production_accuracy_claim':False};(out/'COMPLETE.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='drawn_frames'},indent=2))
if __name__=='__main__':main()
