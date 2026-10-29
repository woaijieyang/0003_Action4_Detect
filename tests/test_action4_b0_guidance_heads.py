"""CPU engineering fixtures, not dev accuracy or authorized formal training."""
import copy, unittest
from dataclasses import replace
import torch
from torch import nn
from model.action4_box_refinement import BoxRefinement
from model.action4_b0_guidance_heads import *
from model.action4_guidance_contract import EvidenceMatch, LocalEvidence
from model.action4_local_evidence import LocalEvidenceBatch
from model.training.action4_supervision import weighted_objective, step_if_supervised, clear_unsupervised_heads

class GuidanceHeadsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(4107)
        self.refiner=BoxRefinement();self.head=B0GuidanceHeads()
        self.ids=("real-style:grid:1","yolo:2","yolo:3","yolo:4","yolo:5")
        self.pooled=torch.randn(5,2304,requires_grad=True)
        self.geom=torch.randn(5,4,requires_grad=True)
        self.boxes=torch.tensor([[10.,10,30,30]]*5)
        self.routes=torch.tensor([False,True,True,True,True])
        self.priors=torch.tensor([[.8,.7,.56]])
        e=LocalEvidence(text_positive=EvidenceMatch(.8,True,"matched","unique",("yolo-support",)))
        self.evidence=LocalEvidenceBatch(self.ids,(e,)+(LocalEvidence(),)*4,())
    def forward(self,evidence=None):
        return self.head(self.refiner,self.pooled,self.geom,self.boxes,candidate_ids=self.ids,
            evidence=self.evidence if evidence is None else evidence,yolo_only=self.routes,
            b0_priors=self.priors,b0_prior_ids=(self.ids[0],))
    def targets(self):
        gt=torch.full((5,4),float("nan"));gt[1]=torch.tensor([12.,11,35,33])
        return GuidanceTargets(self.ids,torch.tensor([BACKGROUND,POSITIVE,MIXED,IGNORE,UNKNOWN]),
            torch.tensor([False,True,False,False,False]),
            torch.tensor([False,True,False,False,False]),gt)
    def test_shared_appearance_not_registered_twice_and_b0_unchanged(self):
        self.assertFalse(any("appearance" in k for k in self.head.state_dict()))
        self.assertFalse(set(map(id,self.head.parameters())) & set(map(id,self.refiner.parameters())))
        o=self.forward()
        b,q,d=self.refiner(self.pooled[:1],self.geom[:1],self.priors,self.boxes[:1])
        torch.testing.assert_close(o.full_boxes[:1],b)
        torch.testing.assert_close(o.quality_logits[:1],q)
        torch.testing.assert_close(o.deltas[:1],d)
    def test_six_features_actually_change_identity_logits_only(self):
        empty=LocalEvidenceBatch(self.ids,(LocalEvidence(),)*5,())
        with torch.no_grad():
            a=self.forward();b=self.forward(empty)
        self.assertNotEqual(float(a.identity_logits[0]),float(b.identity_logits[0]))
        torch.testing.assert_close(a.full_boxes,b.full_boxes)
        torch.testing.assert_close(a.quality_logits,b.quality_logits)
    def test_missing_external_evidence_does_not_define_negative_label(self):
        empty=LocalEvidenceBatch(self.ids,(LocalEvidence(),)*5,())
        loss=guidance_loss_sums(self.forward(empty),self.targets())
        self.assertEqual(loss.counts["identity_positive"],1)
        self.assertEqual(loss.counts["identity_negative"],1)
        self.assertEqual(loss.counts["full_box_regression"],1)
        self.assertEqual(loss.counts["full_box_quality"],1)
        self.assertTrue(torch.isnan(loss.quality_targets[2:]).all())
    def test_identity_reordering_and_fake_yolo_priors_reject(self):
        with self.assertRaisesRegex(ValueError,"IDs/order"):
            self.forward(replace(self.evidence,candidate_ids=self.ids[::-1]))
        with self.assertRaisesRegex(ValueError,"B0 priors"):
            self.head(self.refiner,self.pooled,self.geom,self.boxes,candidate_ids=self.ids,
                evidence=self.evidence,yolo_only=self.routes,b0_priors=self.priors,b0_prior_ids=("wrong",))
        with self.assertRaisesRegex(ValueError,"fabricated"):
            self.head(self.refiner,self.pooled,self.geom,self.boxes,candidate_ids=self.ids,
                evidence=self.evidence,yolo_only=torch.ones(5,dtype=torch.bool),b0_priors=torch.empty(0,3))
    def test_selects_eligible_GT_before_arithmetic(self):
        o=self.forward();labels=self.targets()
        loss=guidance_loss_sums(o,labels)
        self.assertTrue(all(torch.isfinite(v) for v in loss.sums.values()))
        corrupt=labels.full_boxes.clone();corrupt[1]=float("nan")
        with self.assertRaisesRegex(ValueError,"Illegal eligible GT"):
            guidance_loss_sums(o,replace(labels,full_boxes=corrupt))
        with self.assertRaisesRegex(ValueError,"ID/order"):
            guidance_loss_sums(o,replace(labels,candidate_ids=self.ids[::-1]))
    def test_quality_target_detached_from_current_box(self):
        o=self.forward();loss=guidance_loss_sums(o,self.targets())
        self.assertFalse(loss.quality_targets.requires_grad)
        grad=torch.autograd.grad(loss.sums["full_box_quality"],(o.full_boxes,o.deltas),allow_unused=True,retain_graph=True)
        self.assertTrue(all(g is None or not g.any() for g in grad))
        loss.sums["full_box_quality"].backward()
        self.assertGreater(float(self.head.yolo_quality.weight.grad.norm()),0)
        self.assertTrue(self.head.yolo_delta.weight.grad is None or not self.head.yolo_delta.weight.grad.any())
    def test_two_updates_fullhead_gradient_and_strict_reload(self):
        model=nn.ModuleDict({"refiner":self.refiner,"guidance":self.head})
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
        sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1)
        before=self.head.yolo_delta.weight.detach().clone()
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            o=self.forward();local=guidance_loss_sums(o,self.targets())
            total=weighted_objective(local.sums,local.counts,{"identity_positive":.5,"identity_negative":.5},1)
            total.backward()
            self.assertGreater(float(self.head.yolo_delta.weight.grad.norm()),0)
            self.assertGreater(float(self.refiner.appearance.appearance[1].weight.grad.norm()),0)
            did,_=step_if_supervised(model,opt,sch,local.counts,{})
            self.assertTrue(did)
        self.assertFalse(torch.equal(before,self.head.yolo_delta.weight))
        state=copy.deepcopy(model.state_dict())
        other=nn.ModuleDict({"refiner":BoxRefinement(),"guidance":B0GuidanceHeads()})
        result=other.load_state_dict(state,strict=True)
        self.assertFalse(result.missing_keys);self.assertFalse(result.unexpected_keys)
        with torch.no_grad():
            expected=self.forward()
            actual=other["guidance"](other["refiner"],self.pooled,self.geom,self.boxes,candidate_ids=self.ids,
                evidence=self.evidence,yolo_only=self.routes,b0_priors=self.priors,b0_prior_ids=(self.ids[0],))
        torch.testing.assert_close(actual.identity_logits,expected.identity_logits,rtol=0,atol=0)
        torch.testing.assert_close(actual.full_boxes,expected.full_boxes,rtol=0,atol=0)
        # After momentum exists, all-unknown supervision must skip AdamW and scheduler.
        unknown=GuidanceTargets(self.ids,torch.full((5,),UNKNOWN),torch.zeros(5,dtype=torch.bool),
            torch.zeros(5,dtype=torch.bool),torch.full((5,4),float("nan")))
        opt.zero_grad(set_to_none=True)
        empty=guidance_loss_sums(self.forward(),unknown)
        self.assertEqual(sum(empty.counts.values()),0)
        total=weighted_objective(empty.sums,empty.counts,{},1);total.backward()
        old={k:v.clone() for k,v in model.state_dict().items()};old_epoch=sch.last_epoch
        did,_=step_if_supervised(model,opt,sch,empty.counts,{})
        self.assertFalse(did);self.assertEqual(sch.last_epoch,old_epoch)
        self.assertTrue(all(torch.equal(old[k],v) for k,v in model.state_dict().items()))
    def test_b0_supervision_does_not_authorize_yolo_head_updates(self):
        model=nn.ModuleDict({"refiner":self.refiner,"guidance":self.head})
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.1)
        sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1)
        # Establish nonzero optimizer moments on YOLO heads first.
        initial=guidance_loss_sums(self.forward(),self.targets())
        weighted_objective(initial.sums,initial.counts,{},1).backward()
        step_if_supervised(model,opt,sch,initial.counts,{})
        opt.zero_grad(set_to_none=True)
        labels=GuidanceTargets(self.ids,torch.tensor([BACKGROUND,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN]),
            torch.zeros(5,dtype=torch.bool),torch.zeros(5,dtype=torch.bool),torch.full((5,4),float("nan")))
        loss=guidance_loss_sums(self.forward(),labels)
        self.assertEqual(loss.route_counts["b0"]["full_box_quality"],0)
        self.assertEqual(sum(loss.route_counts["yolo_only"].values()),0)
        weighted_objective(loss.sums,loss.counts,{},1).backward()
        before={k:v.clone() for k,v in self.head.state_dict().items() if k.startswith("yolo_")}
        # This test is one complete single-rank update, so local==global.
        global_update_counts=dict(loss.counts)
        clear_unsupervised_heads(self.head,global_update_counts,{},guidance_head_dependencies())
        step_if_supervised(model,opt,sch,loss.counts,{})
        self.assertTrue(all(torch.equal(v,self.head.state_dict()[k]) for k,v in before.items()))

    def test_full_losses_only_yolo_and_original_b0_loss_unchanged(self):
        from model.action4_box_refinement import loss as original_b0_loss
        routes=torch.tensor([False,False,True,True,True])
        priors=self.priors.expand(2,-1).clone()
        output=self.head(self.refiner,self.pooled,self.geom,self.boxes,candidate_ids=self.ids,
            evidence=self.evidence,yolo_only=routes,b0_priors=priors,b0_prior_ids=self.ids[:2])
        gt=torch.tensor([[12.,11,35,33]]*5)
        labels=GuidanceTargets(self.ids,torch.tensor([POSITIVE,BACKGROUND,POSITIVE,BACKGROUND,MIXED]),
            torch.tensor([True,False,True,False,False]),torch.tensor([True,False,True,False,False]),gt)
        original_outputs=self.refiner(self.pooled[:2],self.geom[:2],priors,self.boxes[:2])
        before=original_b0_loss(original_outputs,self.boxes[:2],gt[:2],torch.tensor([True,False]),torch.ones(2,dtype=torch.bool))
        new=guidance_loss_sums(output,labels)
        self.assertEqual(new.counts,dict(identity_positive=2,identity_negative=2,full_box_regression=1,full_box_giou=1,full_box_quality=2))
        self.assertEqual(new.route_counts["b0"]["full_box_regression"],0)
        self.assertEqual(new.route_counts["b0"]["full_box_quality"],0)
        after=original_b0_loss(original_outputs,self.boxes[:2],gt[:2],torch.tensor([True,False]),torch.ones(2,dtype=torch.bool))
        for k in before:torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)
    def test_zero_branch_connected_and_global_counts_control_clear(self):
        # Rank0 has B0 rows only; rank1 has a supervised YOLO positive.
        routes=torch.zeros(5,dtype=torch.bool)
        rank0=self.head(self.refiner,self.pooled,self.geom,self.boxes,candidate_ids=self.ids,
            evidence=self.evidence,yolo_only=routes,b0_priors=self.priors.expand(5,-1).clone(),b0_prior_ids=self.ids)
        bg=GuidanceTargets(self.ids,torch.zeros(5,dtype=torch.long),torch.zeros(5,dtype=torch.bool),
            torch.zeros(5,dtype=torch.bool),torch.full((5,4),float("nan")))
        local0=guidance_loss_sums(rank0,bg)
        weighted_objective(local0.sums,local0.counts,{},1).backward()
        for name,param in self.head.named_parameters():
            if name.startswith("yolo_"):
                self.assertIsNotNone(param.grad,name)
                self.assertEqual(float(param.grad.abs().sum()),0)
        local1=guidance_loss_sums(self.forward(),self.targets())
        global_update={k:local0.counts[k]+local1.counts[k] for k in LOSS_KEYS}
        clear_unsupervised_heads(self.head,global_update,{},guidance_head_dependencies())
        self.assertTrue(all(p.grad is not None for n,p in self.head.named_parameters() if n.startswith("yolo_")))
        # Whole update globally has no YOLO eligible loss: only then clear.
        clear_unsupervised_heads(self.head,local0.counts,{},guidance_head_dependencies())
        self.assertTrue(all(p.grad is None for n,p in self.head.named_parameters() if n.startswith("yolo_")))
    def test_disabled_losses_have_no_counts_or_invalid_GT_arithmetic(self):
        labels=self.targets()
        invalid=labels.full_boxes.clone();invalid[1]=float("nan")
        weights={k:0. for k in LOSS_KEYS}
        loss=guidance_loss_sums(self.forward(),replace(labels,full_boxes=invalid),loss_weights=weights)
        self.assertTrue(all(v==0 for v in loss.counts.values()))
        self.assertTrue(all(v==0 for counts in loss.route_counts.values() for v in counts.values()))
        self.assertTrue(all(torch.isfinite(v) and v.item()==0 for v in loss.sums.values()))
        self.assertFalse(loss.quality_eligible.any())
        with self.assertRaises(ValueError):
            guidance_loss_sums(self.forward(),labels,loss_weights={"unknown":1.})

    def test_nonunique_positive_has_no_fullbox_or_quality_target(self):
        target=replace(self.targets(),unique_match=torch.zeros(5,dtype=torch.bool))
        local=guidance_loss_sums(self.forward(),target)
        self.assertEqual(local.counts["identity_positive"],1)
        self.assertEqual(local.counts["full_box_regression"],0)
        self.assertEqual(local.counts["full_box_quality"],0)
        self.assertTrue(torch.isnan(local.quality_targets[1]))

if __name__=="__main__":unittest.main()
