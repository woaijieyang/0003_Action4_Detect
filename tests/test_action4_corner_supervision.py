"""Bounded CPU synthetic fixtures: contract/gradient evidence, not accuracy."""
import copy
import unittest
import torch
from dataclasses import replace
from model.action4_corner_observation_head import (CornerLocalizationHead,CornerLocalizationOutput,
                                                   SamCornerFeatures,local_corner_uv)
from model.action4_corner_supervision import (CornerObservationHead,CornerObservationOutput,
    build_corner_targets,corner_loss_sums,effective_weights,parameter_loss_dependencies)
from model.action4_sam_feature_adapter import FeatureIdentity
from model.training.action4_supervision import weighted_objective,clear_unsupervised_heads


class CornerSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
    def ids(self):
        return (FeatureIdentity("engineering-fixture","object","roi","corner224"),)
    def target_args(self):
        return dict(uv_native=torch.full((1,8,2),50.),depth=torch.ones(1,8),
            A224_from_native=torch.eye(3)[None],gt_diagonal224=torch.tensor([200.]),
            gt_known=torch.ones(1,8,dtype=torch.bool),ignore=torch.zeros(1,8,dtype=torch.bool),
            identity_known=torch.ones(1,dtype=torch.bool),identity_correct=torch.ones(1,dtype=torch.bool),
            image_evidence=torch.ones(1,dtype=torch.bool),active=torch.ones(1,dtype=torch.bool),identities=self.ids())
    def output(self,uv=None,b=2.):
        heat=CornerLocalizationOutput(torch.zeros(1,8,224,224,requires_grad=True),self.ids())
        uv=torch.full((1,8,2),50.) if uv is None else uv
        return CornerObservationOutput(heat,uv.clone().requires_grad_(True),
            torch.zeros(1,8,requires_grad=True),torch.full((1,8),b,requires_grad=True),
            torch.zeros(1,8,requires_grad=True))

    def test_hidden_legal_outside_negative_unknown_ignore_and_safe_nan(self):
        args=self.target_args()
        # No visibility flag: an occluded canonical vertex remains valid.
        args["uv_native"][0,1]=torch.tensor([224.,40.])
        args["depth"][0,2]=-1.
        args["gt_known"][0,3]=False;args["uv_native"][0,3]=float("nan")
        args["ignore"][0,4]=True
        args["uv_native"][0,5]=float("nan")
        t=build_corner_targets(**args)
        self.assertEqual(t.valid[0].tolist(),[True,False,False,False,False,False,True,True])
        self.assertEqual(t.known[0].tolist(),[True,True,True,False,False,False,True,True])
        stats=corner_loss_sums(self.output(),t,epoch=6)
        self.assertEqual(stats.counts["corner_heatmap_kl"],3)
        self.assertEqual(stats.counts["corner_valid_bce"],5)
        self.assertEqual(stats.quality_target[0].tolist(),[1.,0.,0.,0.,0.,0.,1.,1.])
        self.assertTrue(all(torch.isfinite(v) for v in stats.sums.values()))
        args=self.target_args();args["image_evidence"][:]=False;args["uv_native"][:]=float("nan")
        neg=build_corner_targets(**args);self.assertEqual(int(neg.known.sum()),8);self.assertFalse(neg.valid.any())
        stats=corner_loss_sums(self.output(),neg,epoch=6)
        self.assertEqual(stats.counts["corner_laplace_nll"],0)
        self.assertEqual(stats.counts["corner_quality_bce"],8);self.assertFalse(stats.quality_target.any())
        args["identity_known"][:]=False
        unknown=build_corner_targets(**args)
        stats=corner_loss_sums(self.output(),unknown,epoch=6)
        self.assertTrue(all(n==0 for n in stats.counts.values()))
        sum(stats.sums.values()).backward()

    def test_affine224_and_no_coordinate_clipping(self):
        args=self.target_args();args["A224_from_native"]=torch.tensor([[[2.,0.,3.],[0.,2.,5.],[0.,0.,1.]]])
        t=build_corner_targets(**args)
        torch.testing.assert_close(t.uv224[0,0],torch.tensor([103.,105.]))
        args["uv_native"][0,0]=torch.tensor([111.,50.]) # x225, not clamped to223
        t=build_corner_targets(**args)
        self.assertFalse(t.valid[0,0]);self.assertTrue(t.known[0,0])

    def test_laplace_units_quality_boundary_detach_and_epoch_weights(self):
        args=self.target_args();t=build_corner_targets(**args)
        uv=torch.full((1,8,2),50.);uv[:,:,0]+=3.;uv[:,:,1]+=4.
        out=self.output(uv)
        stats=corner_loss_sums(out,t,epoch=6)
        expected=8*(7/2+2*torch.tensor(2.).log())
        torch.testing.assert_close(stats.sums["corner_laplace_nll"],expected)
        args["gt_diagonal224"]=torch.tensor([400.])
        other=corner_loss_sums(out,build_corner_targets(**args),epoch=6)
        torch.testing.assert_close(stats.sums["corner_laplace_nll"],other.sums["corner_laplace_nll"])
        self.assertLess(float(other.sums["corner_coordinate"].detach()),float(stats.sums["corner_coordinate"].detach()))
        uv=torch.full((1,8,2),50.);uv[0,0,0]+=2.;uv[0,1,0]+=2.001
        out=self.output(uv);stats=corner_loss_sums(out,t,epoch=6)
        self.assertEqual(stats.quality_target[0,:2].tolist(),[1.,0.])
        stats.sums["corner_quality_bce"].backward()
        self.assertIsNone(out.uv224.grad);self.assertIsNone(out.localization.logits.grad)
        for epoch in range(1,6):self.assertEqual(effective_weights(epoch)["corner_quality_bce"],0)
        self.assertEqual(effective_weights(6)["corner_quality_bce"],.2)
        with self.assertRaises(ValueError):effective_weights(0)
        with self.assertRaises(ValueError):corner_loss_sums(replace(out,scale_b224=torch.full((1,8),float("nan"))),t,epoch=6)

    def test_negative_only_ownership_preserves_untrained_heatmap_and_scale(self):
        model=CornerObservationHead()
        for p in model.parameters():p.grad=torch.ones_like(p)
        counts={key:0 for key in effective_weights(1)}
        counts.update(corner_valid_bce=8,corner_quality_bce=8)
        clear_unsupervised_heads(model,counts,effective_weights(1),parameter_loss_dependencies())
        self.assertTrue(all(p.grad is None for p in model.localization.heatmap.parameters()))
        self.assertTrue(all(p.grad is None for p in model.scale_head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.quality_head.parameters()))
        self.assertTrue(all(p.grad is not None for p in model.valid_head.parameters()))
        self.assertTrue(all(p.grad is not None for p in model.localization.p5.parameters()))
        owned=[]
        for entry in parameter_loss_dependencies():
            owned.extend(n for n,_ in model.named_parameters() if any(n==p or n.startswith(p+".") for p in entry["parameter_prefixes"]))
        self.assertEqual(len(owned),len(set(owned)))
        self.assertEqual(set(owned),set(dict(model.named_parameters())))

    def test_fpn_two_actual_updates_detached_quality_and_strict_reload(self):
        torch.manual_seed(704)
        # Synthetic enhanced C5 leaf tests the consumer gradient without model loading.
        c5=torch.nn.Parameter(torch.randn(1,2048,14,14)*.1)
        features=SamCornerFeatures(torch.randn(1,256,112,112)*.1,
            torch.randn(1,512,56,56)*.1,torch.randn(1,1024,28,28)*.1,c5,self.ids())
        model=CornerObservationHead()
        context=dict(identities=self.ids(),sam_available=torch.ones(1),mask_quality=torch.tensor([.8]))
        out=model(features,**context)
        t=build_corner_targets(**self.target_args())
        stats=corner_loss_sums(out,t,epoch=6)
        stats.sums["corner_quality_bce"].backward()
        self.assertIsNone(c5.grad)
        self.assertTrue(all(p.grad is None for name,p in model.named_parameters() if not name.startswith("quality_head.")))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.quality_head.parameters()),0.)
        model.zero_grad(set_to_none=True)
        opt=torch.optim.AdamW(list(model.parameters())+[c5],lr=1e-4,weight_decay=.01)
        original_quality=copy.deepcopy(model.quality_head.state_dict())
        old_c5=c5.detach().clone()
        for _ in range(2):
            opt.zero_grad(set_to_none=True);out=model(features,**context)
            stats=corner_loss_sums(out,t,epoch=1)
            weighted_objective(stats.sums,stats.counts,stats.weights,1).backward()
            self.assertIsNotNone(c5.grad);self.assertGreater(float(c5.grad.abs().sum()),0.)
            clear_unsupervised_heads(model,stats.counts,stats.weights,parameter_loss_dependencies())
            self.assertTrue(all(p.grad is None for p in model.quality_head.parameters()))
            opt.step()
        self.assertFalse(torch.equal(c5,old_c5))
        for key,value in original_quality.items():torch.testing.assert_close(model.quality_head.state_dict()[key],value,rtol=0,atol=0)
        # Existing default localization output and state keys are unchanged.
        with torch.no_grad():
            out=model(features,**context)
            default=model.localization(features,identities=self.ids())
            extended,spatial=model.localization(features,identities=self.ids(),return_features=True)
            torch.testing.assert_close(default.logits,extended.logits,rtol=0,atol=0)
            clone=CornerObservationHead();clone.load_state_dict(model.state_dict(),strict=True)
            restored=clone(features,**context)
            for a,b in [(out.localization.logits,restored.localization.logits),(out.uv224,restored.uv224),
                        (out.valid_logits,restored.valid_logits),(out.scale_b224,restored.scale_b224),(out.quality_logits,restored.quality_logits)]:
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertEqual(set(CornerLocalizationHead().state_dict()),set(model.localization.state_dict()))
        self.assertFalse(hasattr(out,"quality_ready"))


if __name__=="__main__":unittest.main(verbosity=2)
