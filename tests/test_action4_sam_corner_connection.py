"""Real parent/backbone/surface/pose plus new FPN; synthetic CPU gradient evidence."""
import json
import os
from pathlib import Path
import unittest
import torch
from torch.nn import functional as F
from model.action4_joint_surface_rgb import JointSurfaceRGB
from model.action4_corner_observation_head import CornerLocalizationHead, local_corner_uv, SamCornerFeatures
from model.action4_sam_feature_adapter import FeatureIdentity, prepare_mask_aux
from tests.test_action4_sam_joint_forward import synthetic_fixture, module_digest, grad_l1, output_equal

ROOT=Path("/data/zy/vln/a-test")
OUT=ROOT/"outputs/action4_sam_corner_connection_preparation_v001"

class SamCornerConnectionTests(unittest.TestCase):
    def test_real_single_backbone_corner_FPN_and_adapter_gradient(self):
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"),"")
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(601)
        model=JointSurfaceRGB(str(ROOT/"models/default/A0.pt"),enable_sam=True).eval()
        model.head.requires_grad_(False)
        corner=CornerLocalizationHead().eval().requires_grad_(False)
        initial_keys=set(model.state_dict())
        frozen_digest=module_digest(model.base)
        rgb,K,cad,masks=synthetic_fixture();rgb,K,cad,masks=rgb[:1],K[:1],cad[:1],masks[:1]
        ids=(FeatureIdentity("synthetic-frame","instance-a","synthetic-roi","roi448"),)
        correct=prepare_mask_aux(masks,torch.ones(1,1,1,1),identities=ids,candidate_overlap=torch.ones(1,1))
        wrong=prepare_mask_aux(masks.flip(-1),torch.ones(1,1,1,1),identities=ids,candidate_overlap=torch.ones(1,1))
        def forward(mask):
            calls=[];seen=[]
            handle=model.base.backbone.register_forward_hook(lambda *a:calls.append(1))
            p5=corner.p5.register_forward_pre_hook(lambda mod,args:seen.append(args[0]))
            try:
                pose,aux,features=model(rgb,K,cad,mask_aux=mask,identities=ids,return_aux=True,return_corner_features=True)
                self.assertIs(features.c5_sam,aux.c5)
                heat=corner(features,identities=ids)
                self.assertIs(seen[0],aux.c5)
            finally:handle.remove();p5.remove()
            self.assertEqual(calls,[1])
            return pose,aux,features,heat
        with torch.no_grad():
            base_pose,base_aux,base_features,base_heat=forward(None)
            correct_pose,_,_,correct_heat=forward(correct)
            wrong_pose,_,_,wrong_heat=forward(wrong)
            self.assertTrue(torch.equal(base_heat.logits,correct_heat.logits))
            self.assertTrue(torch.equal(base_heat.logits,wrong_heat.logits))
            output_equal(self,base_pose,correct_pose)
            output_equal(self,base_pose,wrong_pose)
            normal=model(rgb,K,cad,mask_aux=correct,identities=ids)
            output_equal(self,normal,correct_pose)
        self.assertEqual(set(model.state_dict()),initial_keys)
        self.assertEqual(base_heat.logits.shape,(1,8,224,224))
        # Synthetic fixed-ID Gaussian target is an engineering localization probe, not real GT.
        axis=torch.arange(224,dtype=torch.float32);yy,xx=torch.meshgrid(axis,axis,indexing="ij")
        centers=torch.tensor([[60,60],[160,60],[160,160],[60,160],[75,75],[145,75],[145,145],[75,145]],dtype=torch.float32)
        target=torch.exp(-((xx[None]-centers[:,0,None,None]).square()+(yy[None]-centers[:,1,None,None]).square())/(2*1.5**2))
        target=(target/target.sum((-2,-1),keepdim=True))[None]
        optimizer=torch.optim.AdamW(model.sam_adapter.parameters(),lr=1e-3,weight_decay=0)
        records=[]
        for step in range(2):
            model.zero_grad(set_to_none=True)
            _,aux,_,heat=forward(correct)
            loss=F.kl_div(heat.logits.flatten(2).log_softmax(-1),target.flatten(2),reduction="sum")/8
            loss.backward()
            row={"step":step+1,"localization_KL":float(loss.detach()),"projection_grad_L1":grad_l1(model.sam_adapter.projection),
                 "encoder_grad_L1":grad_l1(model.sam_adapter.encoder),"quality_grad_L1":grad_l1(model.sam_adapter.quality_head)}
            self.assertGreater(row["projection_grad_L1"],0)
            self.assertEqual(row["quality_grad_L1"],0)
            self.assertEqual(row["encoder_grad_L1"],0) if step==0 else self.assertGreater(row["encoder_grad_L1"],0)
            self.assertTrue(all(p.grad is None for p in model.base.parameters()))
            optimizer.step();records.append(row)
        with torch.no_grad():
            correct_pose,_,features,correct_heat=forward(correct)
            wrong_pose,_,_,wrong_heat=forward(wrong)
            missing_pose,_,_,missing_heat=forward(None)
        self.assertTrue(torch.equal(missing_heat.logits,base_heat.logits))
        output_equal(self,missing_pose,base_pose)
        differences={"correct_vs_wrong_logits":float((correct_heat.logits-wrong_heat.logits).abs().max()),
                     "correct_vs_missing_logits":float((correct_heat.logits-missing_heat.logits).abs().max()),
                     "correct_vs_missing_UV":float((local_corner_uv(correct_heat)-local_corner_uv(missing_heat)).abs().max()),
                     "correct_vs_missing_pose_corners":float((correct_pose["corners"]-missing_pose["corners"]).abs().max())}
        for name,value in differences.items():self.assertGreater(value,0,name)
        self.assertEqual(module_digest(model.base),frozen_digest)
        self.assertEqual(set(model.state_dict()),initial_keys)
        bad_ids=(FeatureIdentity("synthetic-frame","wrong-instance","synthetic-roi","roi448"),)
        with self.assertRaisesRegex(ValueError,"identity"):corner(features,identities=bad_ids)
        bad=SamCornerFeatures(features.c2[:,:,:111],features.c3,features.c4,features.c5_sam,ids)
        with self.assertRaisesRegex(ValueError,"shape"):corner(bad,identities=ids)
        bad=SamCornerFeatures(features.c2,features.c3,features.c4,features.c5_sam*float("nan"),ids)
        with self.assertRaisesRegex(ValueError,"finiteness"):corner(bad,identities=ids)
        # Strict reload of the independently owned new localization head; not a release bundle.
        path=OUT/"engineering_corner_head.pt"
        torch.save(corner.state_dict(),path)
        reloaded=CornerLocalizationHead().eval()
        reloaded.load_state_dict(torch.load(path,weights_only=True),strict=True)
        with torch.no_grad():
            self.assertTrue(torch.equal(reloaded(features,identities=ids).logits,correct_heat.logits))
        (OUT/"VERIFICATION.json").write_text(json.dumps({
            "scope":"CPU_SYNTHETIC_ENGINEERING_GRADIENT_ONLY","real_parent":"models/default/A0.pt",
            "one_backbone_call_per_joint_forward":True,"surface_and_corner_share_same_enhanced_C5":True,
            "zero_init_equal":True,"missing_mask_exact_RGB_fallback":True,"model_state_keys_unchanged":True,
            "backbone_parameters_buffers_unchanged":True,"updates":records,"differences":differences,
            "strict_localization_head_reload_equal":True,"quality_head_isolated_from_corner_loss":True,
            "formal_training":False,"SAM_prediction_performance":False,"stage4_quality_decoder_bundle_complete":False,
            "cuda_initialized":torch.cuda.is_initialized(),"torch_threads":torch.get_num_threads()},indent=2)+"\n")

if __name__=="__main__":unittest.main(verbosity=2)
