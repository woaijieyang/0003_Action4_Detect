"""Verified provided-video frame/timing identity for material seeding."""
from dataclasses import dataclass,field
from fractions import Fraction
from pathlib import Path
import hashlib,json,os
import numpy as np
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_sam_roi_inputs import ImageFrame,verify_image_frame,array_digest,_digest
_TOKEN=object()

def stat(p):
 s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

@dataclass(frozen=True)
class VideoFrameBinding:
 manifest_path:str
 manifest_sha256:str
 manifest_stat:tuple
 image_path:str
 image_stat:tuple
 image_id:str
 pixel_space_id:str
 source_sha256:str
 variant_sha256:str
 frame_id:int
 pts:int
 time_base:str
 _token:object=field(default=None,init=False,repr=False,compare=False)
 _pid:object=field(default=None,init=False,repr=False,compare=False)
 def __reduce_ex__(self,protocol):raise TypeError('Revalidate video binding in each process')


def load_video_frame(manifest_path,manifest_sha256,frame_id):
 if type(frame_id) is not int or frame_id<0:raise ValueError('Nonnegative source frame ID required')
 _digest(manifest_sha256);mp=contained_path(R,manifest_path,exists=True);ms=stat(mp);raw=mp.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=manifest_sha256 or stat(mp)!=ms:raise ValueError('Video manifest identity changed')
 m=json.loads(raw);rows=m['frames'];source=m['source'];_digest(source['source_sha256'])
 if m.get('complete') is not True or len({r['frame_id'] for r in rows})!=len(rows):raise ValueError('Complete unique video manifest required')
 if any(type(r['frame_id']) is not int or type(r['pts']) is not int for r in rows) or any(b['frame_id']<=a['frame_id'] or b['pts']<=a['pts'] for a,b in zip(rows,rows[1:])):raise ValueError('Source frame/time ordering invalid')
 tb=str(Fraction(source['time_base']))
 if Fraction(tb)<=0:raise ValueError('Positive source timebase required')
 row=next((r for r in rows if r['frame_id']==frame_id),None)
 if row is None:raise ValueError('Frame ID absent from source manifest')
 ip=contained_path(R,row['image'],exists=True);ims=stat(ip);data=ip.read_bytes()
 if hashlib.sha256(data).hexdigest()!=row['image_sha256'] or stat(ip)!=ims:raise ValueError('Video RGB source changed')
 import io
 with Image.open(io.BytesIO(data)) as image:rgb=np.asarray(image.convert('RGB')).copy()
 if rgb.shape!=(source['height'],source['width'],3):raise ValueError('Provided-video geometry changed')
 image_id=f"video:{source['source_sha256']}:frame:{frame_id}"
 space='provided_video_native:'+manifest_sha256
 frame=verify_image_frame(ImageFrame(image_id,rgb,space,row['image_sha256'],'native_original',array_digest(rgb)))
 b=VideoFrameBinding(str(mp.relative_to(R)),manifest_sha256,ms,str(ip.relative_to(R)),ims,image_id,space,frame.source_sha256,frame.variant_sha256,frame_id,row['pts'],tb)
 object.__setattr__(b,'_token',_TOKEN);object.__setattr__(b,'_pid',os.getpid());return frame,b


def validate_video_binding(binding,frame,frame_id,pts,time_base):
 if not isinstance(binding,VideoFrameBinding) or binding._token is not _TOKEN or binding._pid!=os.getpid():raise ValueError('Verified video binding required')
 f=verify_image_frame(frame)
 if (binding.image_id,binding.pixel_space_id,binding.source_sha256,binding.variant_sha256)!=(f.image_id,f.pixel_space_id,f.source_sha256,f.variant_sha256):raise ValueError('Video binding belongs to another RGB/frame domain')
 if (binding.frame_id,binding.pts,binding.time_base)!=(frame_id,pts,str(Fraction(time_base))):raise ValueError('Source frame/PTS/timebase mismatch')
 if stat(contained_path(R,binding.manifest_path,exists=True))!=binding.manifest_stat or stat(contained_path(R,binding.image_path,exists=True))!=binding.image_stat:raise ValueError('Bound video source files changed')
 return {'manifest_sha256':binding.manifest_sha256,'image_path':binding.image_path,'frame_id':binding.frame_id,'pts':binding.pts,'time_base':binding.time_base}
