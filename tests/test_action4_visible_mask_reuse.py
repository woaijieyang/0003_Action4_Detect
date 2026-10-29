"""Focused item-local mask reuse tests; real parity opt-in uses existing train inputs."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from data import action4_online_surface_targets as current

ROOT=Path(os.environ["ACTION4_PROJECT_ROOT"])
OUT=ROOT/"outputs/action4_a0_loader_acceleration_v001"
FROZEN=ROOT/"runtime/action4_runs/p2_a0_parallel80_eval10_v001/source"
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
def exact(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b
class MaskReuse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUT.mkdir(exist_ok=True)
        cls.old=module("frozen_surface_mask_reference",FROZEN/"src/data/action4_online_surface_targets.py")
    def test_rounding_origin_edges_empty_and_crop_parity(self):
        with tempfile.TemporaryDirectory(dir=OUT) as folder:
            path=Path(folder)/"mask.png"
            arr=np.zeros((53,71),dtype=np.uint8);arr[2:49:3,1:65:2]=255;arr[20:25,20:40]=1
            Image.fromarray(arr).save(path)
            grids=[np.array([[-40,-20],[-.5,.5],[0,0],[1.5,2.5],[70,52],[71,53],[20,23]],float),
                   np.empty((0,2)),np.array([[-100,-100],[200,200]],float),
                   np.array([[10.5,22.5],[11.5,23.5],[25,24]],float)]
            for origin in [(0,0),(-31,-17),(5,8)]:
                for grid in grids:
                    with self.subTest(origin=origin,grid=grid.tolist()):
                        expected=self.old.sample_binary_mask(str(path),grid,origin,(71,53))
                        with Image.open(path) as image:actual=current._sample_binary_mask_image(image,grid,origin,(71,53))
                        self.assertTrue(exact(expected,actual))
    def test_dimension_and_rgb_validation_unchanged(self):
        with tempfile.TemporaryDirectory(dir=OUT) as folder:
            path=Path(folder)/"rgb.png";Image.new("RGB",(7,5)).save(path)
            for mod in (self.old,current):
                with self.assertRaisesRegex(ValueError,"dimensions"):mod.sample_binary_mask(str(path),np.array([[0,0]]),expected_wh=(8,5))
                with self.assertRaisesRegex(ValueError,"one channel"):mod.sample_binary_mask(str(path),np.array([[0,0]]),expected_wh=(7,5))
                self.assertTrue(np.array_equal(mod.sample_binary_mask(str(path),np.array([[-1,-1]])),np.zeros(1,np.float32)))
    @unittest.skipUnless(os.environ.get("ACTION4_REAL_MASK_REUSE")=="1","explicit existing real-input probe")
    def test_real_frozen_parity_decode_count_and_timing(self):
        torch.set_num_threads(1)
        cfg=json.loads((FROZEN/"resolved_config.json").read_text())
        records=[json.loads(l) for l in (ROOT/"data/phrase2/train/annotations.jsonl").read_text().splitlines()]
        old_pose=module("frozen_pose_mask_reference",FROZEN/"src/data/action4_pose_rgb_batch.py")
        self.old.prepare_pose_rgb=old_pose.prepare_pose_rgb
        before=module("pre_edit_surface_reference",OUT/"BEFORE.py")
        rows=[]
        for idx in (0,1733,3466,5199):
            rec=records[idx];path=ROOT/rec["image"]
            self.assertTrue(path.resolve().is_relative_to(ROOT))
            with Image.open(path) as image:rgb=torch.from_numpy(np.array(image.convert("RGB"))).permute(2,0,1)
            for instance in (0,1):
                item={"record":rec,"instance":instance,"image":rgb}
                for epoch in (0,37):
                    seed=cfg["seed"]+epoch*1000003+(idx*2+instance)*104729
                    args=dict(seeds=[seed],jitter_config=cfg["roi_jitter"],surface_source=cfg["surface_source"])
                    visible=(ROOT/rec["instances"][instance]["mask_visible"]).resolve()
                    times={};counts={};results={}
                    # Alternate order to reduce systematic warm-cache bias.
                    order=[("old",self.old),("new",current)]
                    if (instance+epoch)%2:order.reverse()
                    for name,mod in order:
                        opens=[0];decodes=[0];original=Image.open
                        def tracked(path,*a,**k):
                            image=original(path,*a,**k)
                            if Path(path).resolve()==visible:
                                opens[0]+=1;original_load=image.load
                                def load(*a,**k):
                                    first=getattr(image,"fp",None) is not None
                                    result=original_load(*a,**k)
                                    if first:decodes[0]+=1
                                    return result
                                image.load=load
                            return image
                        start=time.perf_counter()
                        with patch.object(Image,"open",tracked):results[name]=mod.prepare_online_surface_batch([item],"train",**args)
                        times[name]=time.perf_counter()-start;counts[name]={"opens":opens[0],"decodes":decodes[0]}
                    baseline=before.prepare_online_surface_batch([item],"train",**args)
                    self.assertTrue(exact(baseline,results["new"]),f"complete pre-edit parity {idx}/{instance}/{epoch}")
                    # The working pose builder already adds corner_gt224; retain
                    # it via full baseline parity, compare frozen common fields.
                    frozen_view=list(results["new"])
                    frozen_view[3]=[{k:v for k,v in m.items() if k!="corner_gt224"} for m in frozen_view[3]]
                    self.assertTrue(exact(results["old"],tuple(frozen_view)),f"frozen parity {idx}/{instance}/{epoch}")
                    self.assertEqual(counts["old"],{"opens":2,"decodes":2})
                    self.assertEqual(counts["new"],{"opens":1,"decodes":1})
                    rows.append(dict(index=idx,instance=instance,epoch=epoch,seed=seed,times=times,visible_counts=counts,exact=True))
        (OUT/"REAL_PARITY_AND_TIMING.json").write_text(json.dumps({"rows":rows,"GPU_calls":0,"full_training_speedup_measured":False},indent=2)+"\n")
if __name__=="__main__":unittest.main()
