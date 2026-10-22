"""Shared saved-PNG semantic verification using the unchanged renderer function.
Read-only. Neither counts, PNGs, renderer flags nor historical records are repaired.
"""
from pathlib import Path
import hashlib,math,re,sys
import numpy as np
from data.action4_global_render import mask_measurements
ROOT=Path('/data/zy/vln/a-test')
RETAINED_FIELDS=('area_visible_px','area_inframe_amodal_px','area_full_amodal_px','occlusion_inframe','occlusion_inframe_raw','truncation','mask_nonnested_boundary_pixels','visible_area_at_fullinput_px','ignore','ignore_reason')

def _path(name):
 if not isinstance(name,str) or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Mask path must be project-relative without traversal')
 path=(ROOT/name).resolve()
 if not path.is_relative_to(ROOT):raise ValueError('Mask path escapes project')
 return path

def _sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def _decode_mask(path):
 if 'bpy' in sys.modules:
  # Preserve the renderer's bottom-up orientation and red-channel semantics,
  # but enforce the same binary-PNG contract as the host before thresholding.
  import bpy
  image=bpy.data.images.load(str(path),check_existing=False)
  try:
   if image.file_format!='PNG':raise ValueError('Actual PNG mask required')
   width,height=image.size
   pixels=np.empty(width*height*4,dtype=np.float32);image.pixels.foreach_get(pixels)
   red=pixels.reshape(height,width,4)[::-1,:,0]
   if not np.isfinite(red).all() or np.any((red!=0)&(red!=1)):raise ValueError('Source binary mask PNG values required')
   return (red>.5).copy()
  finally:bpy.data.images.remove(image)
 from PIL import Image
 with Image.open(path) as image:
  if image.format!='PNG':raise ValueError('Actual PNG mask required')
  image.load();array=np.asarray(image)
  red=array if array.ndim==2 else array[:,:,0]
  if red.dtype==np.bool_:return red.copy()
  if red.dtype==np.uint8:maximum=255
  elif red.dtype==np.uint16 or (red.dtype==np.int32 and image.mode=='I'):maximum=65535
  else:raise ValueError('Unsupported source binary mask PNG dtype')
  # Source masks are binary; ambiguous gamma/antialias bytes need a new explicit contract.
  if np.any((red!=0)&(red!=maximum)):raise ValueError('Source binary mask PNG values required')
  return red>maximum/2

def _equal(actual,expected):
 if type(expected) is bool:return type(actual) is bool and actual==expected
 if type(expected) is int:return type(actual) is int and actual==expected
 if type(expected) is float:return type(actual) in (int,float) and math.isfinite(actual) and actual==expected
 return type(actual) is type(expected) and actual==expected

def verify_row_masks(row,width,height,*,excluded=False):
 """Recompute spatial tolerance and every stored measurement from immutable pixels."""
 if any(type(v) is not int or v<=0 for v in (width,height)):raise ValueError('Positive integer native image size')
 if row.get('amodal_expanded_shape_hw')!=[3*height,3*width] or row.get('amodal_expanded_origin_px')!=[-width,-height]:raise ValueError('Exact original expanded canvas and origin')
 paths={};digests={};arrays={}
 for key in ('mask_visible','mask_amodal_expanded'):
  path=_path(row.get(key));digest=row.get(key+'_sha256')
  if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest) or _sha(path)!=digest:raise ValueError('Immutable PNG SHA mismatch: '+key)
  paths[key]=path;digests[key]=digest;arrays[key]=_decode_mask(path)
 measurements=mask_measurements(arrays['mask_visible'],arrays['mask_amodal_expanded'],width,height)
 expected_excluded=measurements.get('excluded',False)
 if expected_excluded is not excluded:raise ValueError('PNG-derived outside_image status differs from retained/excluded row')
 if excluded:
  if any(k in row for k in RETAINED_FIELDS if k!='area_full_amodal_px') or 'bbox_visible_xyxy' in row:raise ValueError('Outside-image row cannot carry retained measurements')
  keys=('reason','area_full_amodal_px')
 else:keys=RETAINED_FIELDS
 for key in keys:
  if key not in row or not _equal(row[key],measurements[key]):raise ValueError('Stored measurement differs from PNG recomputation: '+key)
 if not excluded:
  if row.get('bbox_visible_xyxy')!=measurements.get('bbox_visible_xyxy'):raise ValueError('Stored visible bbox differs from actual PNG support')
 for key,path in paths.items():
  if _sha(path)!=digests[key]:raise ValueError('PNG changed during semantic validation')
 return measurements

def verify_record_masks(record):
 """Shared host/diagnostic gate. Spatial mismatch cannot hide behind plausible counts."""
 width,height=record.get('width'),record.get('height');measurements={};files=[]
 for excluded,key in [(False,'instances'),(True,'excluded_instances')]:
  rows=record.get(key,[])
  if not isinstance(rows,list):raise ValueError('Explicit retained/excluded mask rows')
  for row in rows:
   ident=row.get('instance_id')
   if type(ident) is not int or ident<=0 or str(ident) in measurements:raise ValueError('Unique original positive instance IDs')
   measurements[str(ident)]=verify_row_masks(row,width,height,excluded=excluded)
   files.extend({'path':row[k],'sha256':row[k+'_sha256']} for k in ('mask_visible','mask_amodal_expanded'))
 return {'status':'PASS_PNG_RECOMPUTED_ORIGINAL_MASK_SEMANTICS','measurements_by_instance_id':measurements,'files':files,'boundary_tolerance':'unchanged mask_measurements: max(4,0.005*inframe_amodal_area)','support_thresholds':{'visible_fraction':.1,'visible_fullinput_pixels':16},'raw_measurements_preserved':True,'input_records_modified':False,'actual_pixel_masks_recomputed':True}