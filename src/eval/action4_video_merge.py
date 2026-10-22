"""Pure frame identity and merge checks; model outputs are never repaired."""
import math

def validate_record(row,frame,source,model_sha,config_sha):
 for key in ('frame_id','pts','duration_pts'):
  if row.get(key)!=frame[key] or type(row.get(key)) is not int:raise ValueError('Frame identity/timing mismatch: '+key)
 expected={'time_base':source['time_base'],'source_image':frame['image'],'source_image_sha256':frame['image_sha256'],'checkpoint_sha256':model_sha,'inference_config_sha256':config_sha,'claim_scope':'model_raw_predictions_pending_visual_review'}
 if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('Source/model/config provenance mismatch')
 values=row.get('instances')
 if not isinstance(values,list) or len(values)>2:raise ValueError('At most two predicted objects required')
 ids=set()
 for p in values:
  if p.get('instance_id') in ids:raise ValueError('Duplicate prediction ID')
  ids.add(p.get('instance_id'))
  points=p.get('corners_px')
  if not isinstance(points,list) or len(points)!=8 or any(not isinstance(q,list) or len(q)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in q) for q in points):raise ValueError('Finite ordered8 required')
  box=p.get('bbox_full_xyxy')
  if not isinstance(box,list) or len(box)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in box) or box[2]<=box[0] or box[3]<=box[1]:raise ValueError('Finite orderedbox required')
  if type(p.get('score')) not in (int,float) or not math.isfinite(p['score']) or not 0<=p['score']<=1:raise ValueError('Valid score required')
 return row

def merge_records(groups,manifest,model_sha,config_sha):
 rows={}
 for group in groups:
  for row in group:
   i=row.get('frame_id')
   if type(i) is not int or not 0<=i<len(manifest['frames']) or i in rows:raise ValueError('Duplicate/outofrange frameID')
   rows[i]=validate_record(row,manifest['frames'][i],manifest['source'],model_sha,config_sha)
 if set(rows)!=set(range(manifest['source']['frame_count'])):raise ValueError('Incomplete full frame coverage')
 return [rows[i] for i in range(len(rows))]
