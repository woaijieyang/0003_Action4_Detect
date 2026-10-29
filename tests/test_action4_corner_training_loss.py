"""Existing trainer loss/count integration; CPU engineering fixtures, no run admission."""
import copy
import unittest
import torch
import numpy as np
from data import DataAdapter
from model.action4_corner_observation_head import CornerLocalizationOutput
from model.action4_corner_supervision import CornerObservationOutput,corner_loss_sums,LOSS_WEIGHTS
from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux
from model.training.action4_corner_targets import corner_targets_from_items,corner_target_counts
from model.training.action4_supervision import local_counts,weighted_objective,step_if_supervised
from model.training.action4_stage2_core import loss_sums_on_batch,update_loss_weights,optimizer_for,validate_run_gate
from tests import test_action4_corner_model_wiring as model_fixture
from tests.test_action4_sam_joint_forward import module_digest


def unit(number=0):
    A,_=DataAdapter.letterbox_affine((448,448),(112,112))
    A224,_=DataAdapter.letterbox_affine((448,448),(224,224))
    identity=FeatureIdentity("engineering-"+str(number),"object","roi","roi448")
    aux=MaskAux(torch.zeros(1,1,448,448),torch.zeros(1,1,448,448),torch.zeros(1,1,1,1),
                torch.zeros(1,1),torch.zeros(1,dtype=torch.bool),(identity,))
    return {"sam_input":{"training_gt_instance_id":"GT0","prediction_id":"object",
            "association":{"image_id":identity.image_id,"gt_instance_id":"GT0","status":"matched","prediction_id":"object"}},
        "active":True,"roi":torch.zeros(3,448,448),"sam_enabled":True,"identity":identity,"mask_aux":aux,
        "gt_identity":identity,"gt_visible448":torch.zeros(1,448,448),"sam_quality_supervised":True,
        "surface_masks":torch.zeros(2,112,112),
        "surface_sparse":{"xy_hm":torch.empty(0,2),"xyz":torch.empty(0,3),"regions":torch.empty(0,dtype=torch.long)},
        "geometry":{"visible":torch.tensor([True]*4+[False]*4)},
        "metadata":{"sample_id":identity.image_id,"instance_id":"GT0","affine_hm":A.tolist(),"target_visible_input_support_count":1,"target_identity_valid":True,
            "gt_geometry_valid":True,"corner_gt224":{"schema":"action4.training_corner_gt224.v1","uv_space":"native_integer_centers",
                "image_id":identity.image_id,"gt_instance_id":"GT0",
                "native_uv":[[100.,100.]]*8,"physical_depth_m":[1.]*8,"A224_from_native":A224.tolist(),
                "full_box_diagonal224":150.,"gt_known":[True]*8,"ignore":[False]*8}}}


class CornerTrainingLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_sameROI_native_metadata_and_112_substitution_rejected(self):
        from tests import test_action4_training_crop_specs as fixtures
        from data.action4_pose_rgb_batch import prepare_pose_rgb
        f=fixtures.CropSpecTests();f.setUp()
        rgb,g,meta=prepare_pose_rgb([f.item],crop_specs=[f.spec()])
        packet=meta[0]["corner_gt224"]
        self.assertEqual(packet["native_uv"],f.item["record"]["instances"][0]["corners_px"])
        expected=(np.asarray(f.item["record"]["corners_3d_mm"])/1000)[:,2]+1
        np.testing.assert_array_equal(packet["physical_depth_m"],expected)
        self.assertAlmostEqual(packet["full_box_diagonal224"],2*meta[0]["diagonal"],places=9)
        item=unit();item["metadata"].update(meta[0])
        item["identity"]=FeatureIdentity(meta[0]["sample_id"],"object","roi","roi448")
        item["sam_input"]={"training_gt_instance_id":str(meta[0]["instance_id"]),"prediction_id":"object",
            "association":{"image_id":meta[0]["sample_id"],"gt_instance_id":str(meta[0]["instance_id"]),"status":"matched","prediction_id":"object"}}
        target=corner_targets_from_items([item])
        A=np.asarray(packet["A224_from_native"])
        expected=DataAdapter.transform_points(packet["native_uv"],A)
        torch.testing.assert_close(target.uv224[0],torch.tensor(expected,dtype=torch.float32),atol=1e-6,rtol=2*torch.finfo(torch.float32).eps)
        bad=copy.deepcopy(item);bad["metadata"]["corner_gt224"]["A224_from_native"]=meta[0]["affine_hm"]
        with self.assertRaisesRegex(ValueError,"actual ROI112"):corner_targets_from_items([bad])

    def test_crossframe_crossGT_packet_and_metadata_swap_rejected(self):
        first=unit(0);other=unit(1)
        # Identical affine and pixel domain cannot substitute source identity.
        other["metadata"]["corner_gt224"]["native_uv"]=[[160.,160.]]*8
        for field in ("packet","metadata"):
            bad=copy.deepcopy(first)
            if field=="packet":bad["metadata"]["corner_gt224"]=copy.deepcopy(other["metadata"]["corner_gt224"])
            else:bad["metadata"]=copy.deepcopy(other["metadata"])
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"identity"):
                corner_targets_from_items([bad])
        other=copy.deepcopy(first)
        other["metadata"]["instance_id"]="GT1"
        other["metadata"]["corner_gt224"]["gt_instance_id"]="GT1"
        other["metadata"]["corner_gt224"]["native_uv"]=[[160.,160.]]*8
        for field in ("packet","metadata"):
            bad=copy.deepcopy(first)
            if field=="packet":bad["metadata"]["corner_gt224"]=copy.deepcopy(other["metadata"]["corner_gt224"])
            else:bad["metadata"]=copy.deepcopy(other["metadata"])
            with self.subTest(field=field),self.assertRaises(ValueError):corner_targets_from_items([bad])
        for field in ("image_id","gt_instance_id"):
            bad=copy.deepcopy(first);bad["metadata"]["corner_gt224"].pop(field)
            with self.assertRaises(ValueError):corner_targets_from_items([bad])
        bad=copy.deepcopy(first);bad.pop("sam_input")
        with self.assertRaises(ValueError):corner_targets_from_items([bad])
        bad=copy.deepcopy(first);bad["sam_input"]["association"]["prediction_id"]="another_prediction"
        with self.assertRaises(ValueError):corner_targets_from_items([bad])
        # Prediction ID is NOT a GT ID: the positive fixture deliberately differs.
        self.assertNotEqual(first["identity"].instance_id,first["metadata"]["instance_id"])
        self.assertEqual(int(corner_targets_from_items([first]).valid.sum()),8)
        unassociated=copy.deepcopy(first)
        unassociated["identity"]=FeatureIdentity(first["identity"].image_id,"training_unassociated:GT0","roi","roi448")
        unassociated["sam_input"]["prediction_id"]=None
        unassociated["sam_input"]["association"].update(status="missing",prediction_id=None)
        self.assertEqual(int(corner_targets_from_items([unassociated]).valid.sum()),8)
        for owner in ("sam_input","association"):
            bad=copy.deepcopy(unassociated)
            if owner=="sam_input":bad["sam_input"].pop("prediction_id")
            else:bad["sam_input"]["association"].pop("prediction_id")
            with self.assertRaises(ValueError):corner_targets_from_items([bad])

    def test_whole_update_counts_uneven_rank_micro_equivalence(self):
        items=[unit(i) for i in range(3)]
        items[1]["metadata"]["target_visible_input_support_count"]=0 # known negative
        items[2]["metadata"]["corner_gt224"]["gt_known"]=[False]*8
        items[2]["metadata"]["corner_gt224"]["native_uv"]=[[float("nan")]*2]*8
        full=local_counts(items,sam_enabled=True,corner_enabled=True)
        self.assertEqual(full["corner_heatmap_kl"],8)
        self.assertEqual(full["corner_valid_bce"],16)
        rankcounts=[local_counts(items[:2],sam_enabled=True,corner_enabled=True),
                    local_counts(items[2:],sam_enabled=True,corner_enabled=True)]
        self.assertEqual(full,{k:sum(c[k] for c in rankcounts) for k in full})
        self.assertEqual(set(rankcounts[0]),set(rankcounts[1]))
        with self.assertRaises(ValueError):local_counts(items,corner_enabled=True)
        axis=torch.arange(224);yy,xx=torch.meshgrid(axis,axis,indexing="ij")
        template=-((xx-49.75).square()+(yy-49.75).square())/5
        def stats(mini,parameter):
            targets=corner_targets_from_items(mini)
            n=len(mini);ids=tuple(x["identity"] for x in mini)
            logits=template[None,None].expand(n,8,-1,-1)*(1+parameter)
            out=CornerObservationOutput(CornerLocalizationOutput(logits,ids),
                torch.full((n,8,2),49.75)+parameter,
                parameter.expand(n,8),torch.ones(n,8)*(1+parameter.sigmoid()),parameter.expand(n,8))
            return corner_loss_sums(out,targets,epoch=6)
        theta=torch.tensor(.2,requires_grad=True)
        whole=stats(items,theta);objective=weighted_objective(whole.sums,full,whole.weights,1)
        objective.backward();reference=theta.grad.clone()
        gradients=[];rankvalues=[]
        partitions=[[items[:1],items[1:2]],[items[2:],[dict(items[2],active=False)]]]
        for partition in partitions:
            parameter=torch.tensor(.2,requires_grad=True);value=0.
            for mini in partition:
                part=stats(mini,parameter)
                weighted=weighted_objective(part.sums,full,part.weights,2)
                weighted.backward();value+=float(weighted.detach())
            gradients.append(parameter.grad);rankvalues.append(value)
        torch.testing.assert_close(sum(gradients)/2,reference,rtol=2e-5,atol=1e-6)
        self.assertAlmostEqual(sum(rankvalues)/2,float(objective.detach()),places=5)

    def test_all_unknown_skip_scheduler_optimizer_and_zero_graph(self):
        item=unit();item["active"]=False
        counts=local_counts([item],sam_enabled=True,corner_enabled=True)
        self.assertTrue(all(v==0 for v in counts.values()))
        model=torch.nn.Linear(1,1);before=copy.deepcopy(model.state_dict())
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1)
        epoch=scheduler.last_epoch
        for p in model.parameters():p.grad=torch.ones_like(p)
        did,norm=step_if_supervised(model,optimizer,scheduler,counts,{k:1 for k in counts})
        self.assertFalse(did);self.assertEqual(scheduler.last_epoch,epoch);self.assertEqual(len(optimizer.state),0)
        for key in before:torch.testing.assert_close(before[key],model.state_dict()[key],rtol=0,atol=0)

    def test_roi_routing_config_rejected_before_model_or_dispatch(self):
        from unittest.mock import patch
        from model.training.action4_stage2_core import construct
        from data.action4_sam_roi_routing import POLICY,validate_config
        base={"role":"A0","mode":"integration","sam_enabled":True,"a0_variant":"a0_sam_image",
              "roi_jitter":copy.deepcopy(POLICY["gt_jitter"]),
              "sam_roi_routing":{"path":"not_read_in_guard","sha256":"a"*64}}
        wrongs=[dict(base,role="B0"),dict(base,sam_enabled=False,a0_variant="a0_baseline"),
                dict(base,mode="full_training"),dict(base,roi_jitter={"center_fraction":0,"scale_range":[1,1]}),
                dict(base,a0_variant="a0_corner_image",corner_enabled=False)]
        for cfg in wrongs:
            with self.subTest(cfg=cfg),patch("model.training.action4_stage2_core.JointSurfaceRGB",side_effect=AssertionError("no model work")),patch("model.training.action4_stage2_core.load_bundle",side_effect=AssertionError("no bundle work")):
                with self.assertRaises(ValueError):construct(cfg,"cpu")
                with self.assertRaises(ValueError):validate_run_gate(cfg,{"launch_allowed":True,"mode":cfg["mode"]})
        validate_config({},"train") # absent optional key stays no-op
        validate_config(base,"train")

    def test_actual_owned_model_two_updates_existing_loss_entry(self):
        # Reuse existing real-parent component constructor; no scheduled training.
        model_fixture.CornerModelWiringTests.setUpClass()
        f=model_fixture.CornerModelWiringTests
        model,cfg=f.model,f.cfg
        item=unit();item.update(roi=f.rgb[0],identity=f.ids[0],gt_identity=f.ids[0],mask_aux=f.aux,
            gt_visible448=f.aux.visible[0].clone(),surface_masks=torch.nn.functional.interpolate(f.aux.visible,size=(112,112),mode="nearest")[0].repeat(2,1,1))
        item["metadata"]["sample_id"]=f.ids[0].image_id
        item["metadata"]["corner_gt224"]["image_id"]=f.ids[0].image_id
        item["sam_input"]={"training_gt_instance_id":"GT0","prediction_id":f.ids[0].instance_id,
            "association":{"image_id":f.ids[0].image_id,"gt_instance_id":"GT0","status":"matched","prediction_id":f.ids[0].instance_id}}
        A,_=DataAdapter.letterbox_affine((448,448),(112,112))
        A224,_=DataAdapter.letterbox_affine((448,448),(224,224))
        R=torch.eye(3);t=torch.tensor([0.,0.,1.]);cad=f.cad[0];camera=cad@R.T+t
        hom=camera@f.K[0].T;corners=hom[:,:2]/camera[:,2:]
        native=DataAdapter.transform_points(corners.numpy(),np.linalg.inv(A))
        item["geometry"]={"K":f.K[0],"canonical":cad,"rotation":R,"translation":t,"corners":corners,
            "diagonal":torch.tensor(100.),"visible":torch.tensor([True]*4+[False]*4)}
        item["surface_sparse"]={"xy_hm":torch.tensor([[30.,40.]]),"xyz":torch.full((1,3),.5),"regions":torch.tensor([1])}
        item["metadata"]["corner_gt224"].update(native_uv=native.tolist(),physical_depth_m=camera[:,2].tolist(),
            A224_from_native=A224.tolist(),full_box_diagonal224=200.)
        item["metadata"]["target_visible_input_support_count"]=100
        frozen=module_digest(model.base)
        opt=optimizer_for(model,cfg);scheduler=torch.optim.lr_scheduler.StepLR(opt,100)
        before=model.corner_head.localization.heatmap.weight.detach().clone()
        before_adapter=model.sam_adapter.projection.weight.detach().clone()
        before_quality=model.corner_head.quality_head[0].weight.detach().clone()
        for epoch in (1,6):
            counts=local_counts([item],sam_enabled=True,corner_enabled=True)
            weights=update_loss_weights(cfg,epoch)
            self.assertEqual(weights["corner_quality_bce"],0. if epoch==1 else .2)
            opt.zero_grad(set_to_none=True);model.train()
            sums=loss_sums_on_batch(model,[item],cfg,"cpu",epoch_one_based=epoch)
            self.assertEqual(set(sums),set(counts));self.assertTrue(all(torch.isfinite(v) for v in sums.values()))
            weighted_objective(sums,counts,weights,1).backward()
            did,_=step_if_supervised(model,opt,scheduler,counts,weights,cfg["independent_head_dependencies"])
            self.assertTrue(did)
        self.assertFalse(torch.equal(before_adapter,model.sam_adapter.projection.weight))
        self.assertFalse(torch.equal(before_quality,model.corner_head.quality_head[0].weight))
        negative=copy.deepcopy(item)
        negative["metadata"]["target_visible_input_support_count"]=0
        negative["surface_sparse"]={"xy_hm":torch.empty(0,2),"xyz":torch.empty(0,3),"regions":torch.empty(0,dtype=torch.long)}
        negative["surface_masks"].zero_()
        negative["gt_visible448"].zero_()
        counts=local_counts([negative],sam_enabled=True,corner_enabled=True)
        weights=update_loss_weights(cfg,6)
        opt.zero_grad(set_to_none=True)
        sums=loss_sums_on_batch(model,[negative],cfg,"cpu",epoch_one_based=6)
        weighted_objective(sums,counts,weights,1).backward()
        from model.training.action4_supervision import clear_unsupervised_heads
        clear_unsupervised_heads(model,counts,weights,cfg["independent_head_dependencies"])
        for head in (model.corner_head.localization.heatmap,model.corner_head.scale_head,model.head.pose):
            self.assertTrue(all(p.grad is None for p in head.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.corner_head.valid_head.parameters()))
        self.assertFalse(torch.equal(before,model.corner_head.localization.heatmap.weight))
        self.assertEqual(module_digest(model.base),frozen)
        with self.assertRaises(ValueError):loss_sums_on_batch(model,[item],cfg,"cpu")
        with self.assertRaisesRegex(ValueError,"scheduled training refused"):validate_run_gate(cfg,{"mode":"integration","launch_allowed":True})


if __name__=="__main__":unittest.main(verbosity=2)
