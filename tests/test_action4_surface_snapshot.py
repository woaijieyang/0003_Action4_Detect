"""CPU engineering contract tests, not accuracy/material acceptance."""
import copy,json,unittest,uuid
from pathlib import Path
from unittest.mock import Mock
from types import SimpleNamespace
import numpy as np
import torch
from data.action4_paths import PROJECT_ROOT as ROOT
from data.action4_sam_roi_inputs import ImageFrame,array_digest
from eval.action4_surface_snapshot import *
from eval.action4_staged_pipeline import StagedPipeline

class Head(torch.nn.Module):
    def forward(self,features,K,canonical):
        n=len(features);return {"surface":{"xyz":torch.full((n,3,112,112),1.5),
            "masks":torch.zeros(n,2,112,112),"regions":torch.zeros(n,65,112,112)},
            "rotation":torch.eye(3).expand(n,-1,-1),"translation":torch.tensor([[0.,0.,1.]]).expand(n,-1),
            "corners":torch.zeros(n,8,2),"camera_corners":torch.ones(n,8,3)}

class SurfaceSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.out=ROOT/"outputs/action4_surface_snapshot_preparation_v001"/("unit_"+uuid.uuid4().hex)
        cls.out.mkdir(parents=True)
    def setUp(self):
        self.rgb=np.arange(48*64*3,dtype=np.uint8).reshape(48,64,3)
        self.frame=ImageFrame("fixture",self.rgb,"native.fixture","a"*64,"original",array_digest(self.rgb))
        self.K=np.array([[90.,0,32],[0,90,24],[0,0,1.]])
        self.cad=np.array([[x,y,z] for x in [-.03,.03] for y in [-.02,.02] for z in [-.01,.01]])
        self.p=StagedPipeline.__new__(StagedPipeline);p=self.p;p.device="cpu";p.a0_kind="joint_pose";p.pose_method="network";p.identity={}
        p.base=SimpleNamespace(backbone=Mock(side_effect=lambda r:torch.zeros(len(r),2048,14,14)))
        p.a0=Head()
        p._detect=Mock(return_value=[dict(score=.9,grid_index=7,bbox_full_xyxy=[2.,3.,28.,29.])])
        p._surface_model_bindings={}
        for role in ("A0","B0"):
            f=self.out/(uuid.uuid4().hex+".bin");f.write_bytes(role.encode())
            p._surface_model_bindings[role]=bind_model_file(f)
    def export(self):
        return self.p.predict(self.rgb,self.K,self.cad,return_surface=True,surface_frame=self.frame)
    def test_same_forward_roundtrip_and_geometry(self):
        default=self.p.predict(self.rgb,self.K,self.cad)
        self.assertNotIn("surface_snapshots",default)
        result=self.export();self.assertEqual(self.p.base.backbone.call_count,2)
        self.assertEqual(default["instances"][0]["corners_px"],result["instances"][0]["corners_px"])
        s=result["surface_snapshots"][0];a=s.arrays
        self.assertTrue((a["xyz_normalized"]==1.5).all())
        np.testing.assert_allclose(a["K112"],a["A112"]@self.K,rtol=2e-6)
        uv=np.array([12.,18.,1.]);np.testing.assert_allclose(np.linalg.inv(a["A112"])@(a["A112"]@uv),uv)
        dest=self.out/(uuid.uuid4().hex+".npz");save_snapshot(dest,s)
        loaded=load_snapshot(dest,frame=self.frame,expected_models=s.metadata["models"])
        for k in a:np.testing.assert_array_equal(a[k],loaded.arrays[k])
        self.assertIsNone(s.metadata["sam_input"])
        self.assertEqual(s.metadata["prediction_id"],"b0:fixture:grid:7")
        with self.assertRaises(FileExistsError):save_snapshot(dest,s)
    def test_wrong_inputs_and_model(self):
        with self.assertRaises(ValueError):self.p.predict(self.rgb,self.K,self.cad,surface_frame=self.frame)
        with self.assertRaises(ValueError):self.p.predict(self.rgb,self.K,self.cad,return_surface=True)
        bad=self.rgb.copy();bad[0,0,0]^=1
        with self.assertRaises(ValueError):self.p.predict(bad,self.K,self.cad,return_surface=True,surface_frame=self.frame)
        f=ROOT/self.p._surface_model_bindings["B0"]["path"];f.write_bytes(b"replaced")
        with self.assertRaisesRegex(ValueError,"changed"):self.export()
    def test_bad_grid_decoder_empty(self):
        self.p._detect.return_value=[dict(score=.9,grid_index=7,prediction_id="b0:wrong:grid:7",bbox_full_xyxy=[2.,3.,28.,29.])]
        with self.assertRaisesRegex(ValueError,"identity"):self.export()
        self.p.pose_method="pnp"
        with self.assertRaisesRegex(ValueError,"network"):self.export()
        self.p.pose_method="network";self.p._detect.return_value=[]
        self.assertEqual(self.export()["surface_snapshots"],[])
        self.assertEqual(self.p.base.backbone.call_count,0)
    def test_strict_metadata_arrays(self):
        s=self.export()["surface_snapshots"][0]
        for key in ("image_id","prediction_id","roi_id"):
            m=copy.deepcopy(s.metadata);m[key]="wrong"
            with self.assertRaises(ValueError):validate_snapshot(SurfaceSnapshot(m,s.arrays))
        for name in ("xyz_normalized","mask_logits","region_logits"):
            a=dict(s.arrays);a[name]=a[name].copy();a[name].flat[0]=np.nan
            with self.assertRaises(ValueError):validate_snapshot(SurfaceSnapshot(s.metadata,a))
        a=dict(s.arrays);a["GT"]=np.ones(1)
        with self.assertRaises(ValueError):validate_snapshot(SurfaceSnapshot(s.metadata,a))
        models=copy.deepcopy(s.metadata["models"]);models["A0"]["sha256"]="b"*64
        with self.assertRaises(ValueError):validate_snapshot(s,expected_models=models)
        with self.assertRaises(ValueError):validate_snapshot(s,expected_prediction_id="b0:other:grid:7")
        with self.assertRaises(ValueError):validate_snapshot(s,expected_roi_id="different")
        dest=self.out/(uuid.uuid4().hex+".npz")
        np.savez(dest,metadata=np.array({},dtype=object))
        with self.assertRaises(ValueError):load_snapshot(dest)

    def test_rehashed_invalid_coordinate_semantics_rejected(self):
        from eval.action4_surface_snapshot import _canonical
        original=self.export()["surface_snapshots"][0]
        def reseal(arrays):
            metadata=copy.deepcopy(original.metadata)
            metadata["roi_id"]=roi_identity(metadata["image_id"],metadata["prediction_id"],arrays["bbox_native"],arrays["A448"])
            metadata["array_sha256"]={k:array_digest(v) for k,v in arrays.items()}
            metadata["receipt"]=_canonical({k:v for k,v in metadata.items() if k!="receipt"})
            return SurfaceSnapshot(metadata,arrays)
        for key in ("A448","A112"):
            for row,col,value in ((2,0,.01),(2,1,.01),(2,2,2.),(0,0,-1.),(1,1,0.),(0,1,.1),(1,0,.1)):
                with self.subTest(key=key,position=(row,col),value=value):
                    arrays={k:v.copy() for k,v in original.arrays.items()}
                    arrays[key][row,col]=value
                    with self.assertRaisesRegex(ValueError,"Axis-aligned"):
                        validate_snapshot(reseal(arrays))
        for key in ("K_native","K112"):
            for row,col,value in ((2,0,.01),(2,1,.01),(2,2,2.),(0,0,0.),(1,1,-1.)):
                with self.subTest(key=key,position=(row,col),value=value):
                    arrays={k:v.copy() for k,v in original.arrays.items()}
                    arrays[key][row,col]=value
                    with self.assertRaisesRegex(ValueError,"Pinhole"):
                        validate_snapshot(reseal(arrays))
        for bbox in ([2.,3.,2.,29.],[28.,3.,2.,29.],[2.,29.,28.,3.]):
            arrays={k:v.copy() for k,v in original.arrays.items()}
            arrays["bbox_native"]=np.asarray(bbox)
            with self.assertRaisesRegex(ValueError,"positive width"):
                validate_snapshot(reseal(arrays))
        # The transport contract deliberately does not pretend raw pose is accepted geometry.
        arrays={k:v.copy() for k,v in original.arrays.items()}
        arrays["translation_m"][2]=-1.
        validate_snapshot(reseal(arrays))
        validate_snapshot(original)

    def test_sam_missing_cache_not_replaced_by_surface_mask(self):
        from eval.action4_sam_pipeline_inputs import SamInferenceContext
        from data.action4_sam_roi_inputs import SOURCE_KEYS
        class EnhancedHead(Head):
            def forward(self,features,K,canonical,**kw):
                return super().forward(features,K,canonical),SimpleNamespace(q_mask=torch.zeros(len(features),1),residual=torch.zeros(len(features),1,1,1))
        self.p.a0_kind="joint_pose_sam";self.p.a0=EnhancedHead()
        models=model_identities(self.p._surface_model_bindings)
        self.p.identity["B0_bundle_sha256"]=models["B0"]["sha256"]
        sources={k:"b"*64 for k in SOURCE_KEYS};sources["b0_model_sha256"]=models["B0"]["sha256"]
        context=SamInferenceContext(self.frame,{},sources)
        result=self.p.predict(self.rgb,self.K,self.cad,return_surface=True,sam_context=context)
        snap=result["surface_snapshots"][0]
        self.assertEqual(snap.metadata["sam_input"]["upstream_status"],"cache_absent")
        self.assertIsNone(snap.metadata["sam_input"]["cache_metadata"])
        self.assertFalse(result["instances"][0]["sam_input"]["available"])

if __name__=="__main__":unittest.main()
