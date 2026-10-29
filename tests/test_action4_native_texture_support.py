import unittest
import numpy as np
from data.action4_native_texture_support import prepare_texture_support
from data.action4_sam_roi_inputs import array_digest

class NativeSupportTests(unittest.TestCase):
    def setup_inputs(self):
        a=np.zeros((8,8),bool);a[1:5,1:5]=True
        b=np.zeros((8,8),bool);b[3:7,3:7]=True
        masks={"a":a,"b":b}
        receipt=dict(frame_id=4,pts=1600,time_base=[1,11988],RGB_sha256="a"*64,
            pixel_space_id="raw_video_native_integer_centers",status="current_native_response",generation=0,
            per_track_status={k:"current_mask_present" for k in masks},mask_sha256={k:array_digest(v) for k,v in masks.items()})
        args=dict(frame_id=4,pts=1600,time_base=[1,11988],rgb_sha256="a"*64,image_shape=(8,8),tracks=["a","b"])
        return masks,receipt,args
    def test_overlap_excludes_pixels_not_bodies(self):
        m,r,a=self.setup_inputs();out,log=prepare_texture_support(m,r,**a)
        self.assertEqual(set(out),{"a","b"})
        self.assertFalse(((out["a"]>0)&(out["b"]>0)).any())
        self.assertEqual(log["a"]["ambiguous_pixels_excluded"],4)
        self.assertEqual(int(m["a"].sum()),16)
    def test_missing_is_zero_not_full_image(self):
        m,r,a=self.setup_inputs();del m["b"]
        out,log=prepare_texture_support(m,r,**a)
        self.assertFalse(out["b"].any());self.assertEqual(log["b"]["status"],"no_texture_observation")
    def test_identity_and_duplicate_rejected(self):
        m,r,a=self.setup_inputs()
        with self.assertRaises(ValueError):prepare_texture_support(m,r,**dict(a,frame_id=5))
        m["b"]=m["a"].copy();r["mask_sha256"]["b"]=array_digest(m["b"])
        with self.assertRaises(ValueError):prepare_texture_support(m,r,**a)
