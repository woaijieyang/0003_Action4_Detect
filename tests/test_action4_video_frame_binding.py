import unittest,json,hashlib,numpy as np
from pathlib import Path
from PIL import Image
from dataclasses import replace
from data.action4_video_frame_binding import load_video_frame,validate_video_binding
from data.action4_sam_roi_inputs import ImageFrame,array_digest
R=Path('/data/zy/vln/a-test');OUT=R/'outputs/action4_surface_anchor_preparation_v001/timing_fixtures'
class VideoTimingTests(unittest.TestCase):
 def setUp(self):
  OUT.mkdir(exist_ok=True);rows=[]
  for i in range(2):
   p=OUT/f'f{i}.png';Image.fromarray(np.full((8,10,3),i*50,np.uint8)).save(p);rows.append({'frame_id':i,'pts':i*400,'image':str(p),'image_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
  self.path=OUT/'manifest.json';self.path.write_text(json.dumps({'complete':True,'source':{'source_sha256':'a'*64,'width':10,'height':8,'time_base':'1/11988'},'frames':rows}));self.sha=hashlib.sha256(self.path.read_bytes()).hexdigest()
 def test_actual_row_bound_and_wrong_frame_time_rejected(self):
  f,b=load_video_frame(str(self.path),self.sha,0);self.assertEqual(validate_video_binding(b,f,0,0,'1/11988')['frame_id'],0)
  for args in [(1,0,'1/11988'),(0,400,'1/11988'),(0,0,'1/30')]:
   with self.assertRaises(ValueError):validate_video_binding(b,f,*args)
  f1,_=load_video_frame(str(self.path),self.sha,1)
  with self.assertRaises(ValueError):validate_video_binding(b,f1,0,0,'1/11988')
 def test_copied_binding_and_source_mutation_rejected(self):
  f,b=load_video_frame(str(self.path),self.sha,0)
  with self.assertRaises(ValueError):validate_video_binding(replace(b,frame_id=1),f,1,0,'1/11988')
  self.path.write_text(self.path.read_text()+' ')
  with self.assertRaises(ValueError):validate_video_binding(b,f,0,0,'1/11988')
 def test_declared_RGB_digest_cannot_replace_source_file_pixels(self):
  f,b=load_video_frame(str(self.path),self.sha,0);pixels=np.ones_like(f.pixels)
  altered=ImageFrame(f.image_id,pixels,f.pixel_space_id,f.source_sha256,f.variant_id,array_digest(pixels))
  with self.assertRaises(ValueError):validate_video_binding(b,altered,0,0,'1/11988')
if __name__=='__main__':unittest.main(verbosity=2)
