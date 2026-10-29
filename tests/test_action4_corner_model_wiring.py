"""CPU component integration only; no scheduler, formal training or pose admission."""
import copy
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from model.action4_joint_surface_rgb import JointSurfaceRGB
from model.action4_release_bundle import load_bundle,SAM_SCHEMA,CORNER_SCHEMA
from model.action4_sam_joint import (SAM_VARIANT,CORNER_VARIANT,CORNER_PROTOCOL,QUALITY_KEY,
    QUALITY_PREFIX,CornerCandidateOutput,validate_corner_config)
from model.action4_sam_feature_adapter import PROTOCOL_ID,prepare_mask_aux
from model.action4_corner_supervision import build_corner_targets,corner_loss_sums,parameter_loss_dependencies,LOSS_WEIGHTS
from model.training.action4_stage2_core import construct,optimizer_for,bundle_snapshot,validate_run_gate
from model.training.action4_supervision import weighted_objective,clear_unsupervised_heads,A0_KEYS
from model.training.action4_runtime import canonical_hash
from tests.test_action4_sam_joint_forward import synthetic_fixture,IDS,module_digest,output_equal

R=Path("/data/zy/vln/a-test")
OUT=R/"outputs/action4_corner_stage4_preparation_v001/model_wiring_fixture"


def sha(p):
    with Path(p).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


class CornerModelWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(718);OUT.mkdir(parents=True,exist_ok=True)
        parent=R/"models/default/A0.pt";original=torch.load(parent,map_location="cpu",weights_only=True)
        cls.default_sha=sha(parent)
        cfg=dict(role="A0",mode="integration",train_backbone=False,parent_bundle=str(parent),
            parent_best_step=original["selected_step"],parent_bundle_sha256=sha(parent),
            surface_lr=1e-5,lr=5e-5,weight_decay=1e-4,sam_adapter_lr=1e-4,sam_quality_lr=1e-4,
            a0_variant=SAM_VARIANT,sam_enabled=True,sam_protocol_id=PROTOCOL_ID,
            independent_head_dependencies=[{"parameter_prefixes":[QUALITY_PREFIX],"losses":[QUALITY_KEY]}],
            loss_weights=dict({k:1. for k in A0_KEYS},sam_mask_quality=.1))
        sam,parent_state=construct(cfg,"cpu")
        cls.sam_state=bundle_snapshot(sam,parent_state,cfg,0)
        cls.sam_path=OUT/"synthetic_SAM_parent.pt";torch.save(cls.sam_state,cls.sam_path)
        del sam,original
        stub=OUT/"synthetic_cache_identity.json";stub.write_text('{"scope":"engineering fixture only"}\n')
        ref={"path":str(stub.relative_to(R)),"sha256":sha(stub)}
        cls.cfg=copy.deepcopy(cfg)
        cls.cfg.update(a0_variant=CORNER_VARIANT,corner_enabled=True,parent_bundle=str(cls.sam_path),
            parent_best_step=0,parent_bundle_sha256=sha(cls.sam_path),sam_adapter_lr=1e-5,
            corner_lr=1e-4,corner_quality_lr=1e-4,corner_protocol=copy.deepcopy(CORNER_PROTOCOL),
            corner_counterpart={"B0":{"path":"models/candidates/action4_improvement/stage1_B0_completed_20261024/B0.pt",
                "sha256":sha(R/"models/candidates/action4_improvement/stage1_B0_completed_20261024/B0.pt")},
                "train_sam_cache":ref,"eval_sam_cache":copy.deepcopy(ref)})
        cls.cfg["loss_weights"].update(LOSS_WEIGHTS)
        cls.cfg["independent_head_dependencies"] += [
            {"parameter_prefixes":["sam_adapter.encoder","sam_adapter.projection"],
             "losses":list(A0_KEYS)+list(LOSS_WEIGHTS)[:-1]}]+parameter_loss_dependencies("corner_head")
        cls.cfg["independent_head_dependencies"] += [
            {"parameter_prefixes":["head.surface"],"losses":list(A0_KEYS)},
            {"parameter_prefixes":["head.pose"],"losses":[k for k in A0_KEYS if k not in ("xyz","region","visible_mask","amodal_mask")]}]
        cls.model,cls.parent=construct(cls.cfg,"cpu")
        rgb,K,cad,masks=synthetic_fixture()
        cls.rgb,cls.K,cls.cad=rgb[:1],K[:1],cad[:1]
        cls.ids=IDS[:1]
        cls.aux=prepare_mask_aux(masks[:1],torch.ones(1,1,1,1),identities=cls.ids,candidate_overlap=torch.ones(1,1))

    def test_owned_forward_two_updates_and_complete_bundle(self):
        model=self.model;cfg=self.cfg;model.train()
        frozen=module_digest(model.base)
        optimizer=optimizer_for(model,cfg)
        owned=[id(p) for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(owned),len(set(owned)))
        self.assertEqual(set(owned),{id(p) for p in model.parameters() if p.requires_grad})
        self.assertEqual({g["name"] for g in optimizer.param_groups},
                         {"surface","pose","sam_adapter","sam_quality","corner_localization","corner_quality"})
        before_head=model.corner_head.localization.heatmap.weight.detach().clone()
        before_adapter=model.sam_adapter.projection.weight.detach().clone()
        targets=build_corner_targets(uv_native=torch.full((1,8,2),75.),depth=torch.ones(1,8),
            A224_from_native=torch.eye(3)[None],gt_diagonal224=torch.tensor([180.]),
            gt_known=torch.ones(1,8,dtype=torch.bool),ignore=torch.zeros(1,8,dtype=torch.bool),
            identity_known=torch.ones(1,dtype=torch.bool),identity_correct=torch.ones(1,dtype=torch.bool),
            image_evidence=torch.ones(1,dtype=torch.bool),active=torch.ones(1,dtype=torch.bool),identities=self.ids)
        for _ in range(2):
            calls=[];hook=model.base.backbone.register_forward_hook(lambda *args:calls.append(1))
            optimizer.zero_grad(set_to_none=True)
            try:out=model(self.rgb,self.K,self.cad,mask_aux=self.aux,identities=self.ids,return_corner=True)
            finally:hook.remove()
            self.assertEqual(calls,[1]);self.assertIsInstance(out,CornerCandidateOutput)
            stats=corner_loss_sums(out.corners,targets,epoch=1)
            weighted_objective(stats.sums,stats.counts,stats.weights,1).backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.sam_adapter.projection.parameters() if p.grad is not None),0)
            clear_unsupervised_heads(model,stats.counts,stats.weights,parameter_loss_dependencies("corner_head"))
            optimizer.step()
        self.assertFalse(torch.equal(before_head,model.corner_head.localization.heatmap.weight))
        self.assertFalse(torch.equal(before_adapter,model.sam_adapter.projection.weight))
        self.assertEqual(module_digest(model.base),frozen)
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))
        model.eval()
        with torch.no_grad():expected=model(self.rgb,self.K,self.cad,mask_aux=self.aux,identities=self.ids,return_corner=True)
        state=bundle_snapshot(model,self.parent,cfg,2)
        self.assertEqual(state["schema"],CORNER_SCHEMA)
        self.assertFalse(state["quality_ready"]);self.assertFalse(state["deployment_admitted"])
        destination=OUT/"research_corner_candidate.pt";torch.save(state,destination)
        with self.assertRaisesRegex(ValueError,"Unadmitted corner"):load_bundle(destination,"A0")
        restored,base,loaded=load_bundle(destination,"A0",allow_corner_candidate=True)
        with torch.no_grad():actual=restored(self.rgb,self.K,self.cad,mask_aux=self.aux,identities=self.ids,return_corner=True)
        output_equal(self,expected.raw_pose,actual.raw_pose)
        for key in ("uv224","valid_logits","scale_b224","quality_logits"):
            torch.testing.assert_close(getattr(expected.corners,key),getattr(actual.corners,key),rtol=0,atol=0)
        torch.testing.assert_close(expected.corners.localization.logits,actual.corners.localization.logits,rtol=0,atol=0)
        self.assertEqual(loaded["corner_counterpart"],cfg["corner_counterpart"])
        # Parent initialization restores every head, rather than creating a new random one.
        restored_parent=JointSurfaceRGB(destination,enable_sam=True,enable_corner=True,parent_bundle_sha256=sha(destination)).eval()
        with torch.no_grad():again=restored_parent(self.rgb,self.K,self.cad,mask_aux=self.aux,identities=self.ids,return_corner=True)
        torch.testing.assert_close(actual.corners.localization.logits,again.corners.localization.logits,rtol=0,atol=0)
        self.assertEqual(sha(R/"models/default/A0.pt"),self.default_sha)

    def test_fail_closed_schema_config_parameters_and_legacy_calls(self):
        state=bundle_snapshot(self.model,self.parent,self.cfg,0)
        for kind in ("head","protocol","schema","ready","config","parent_path","parent_step"):
            bad=dict(state)
            if kind=="head":bad.pop("corner_head")
            elif kind=="protocol":bad.pop("corner_protocol")
            elif kind=="schema":bad["schema"]=SAM_SCHEMA
            elif kind=="ready":bad["quality_ready"]=True
            elif kind=="parent_path":bad["parent_bundle"]="wrong.pt"
            elif kind=="parent_step":bad["parent_selected_step"]+=1
            else:
                bad["config"]=copy.deepcopy(state["config"]);bad["config"].pop("corner_protocol")
                bad["config_id"]=canonical_hash(bad["config"])
            with self.subTest(kind=kind),patch("torch.load",return_value=bad):
                with self.assertRaises(ValueError):load_bundle(self.sam_path,"A0",allow_corner_candidate=True)
        bad=dict(state);bad["corner_head"]=dict(state["corner_head"]);bad["corner_head"].pop(next(iter(bad["corner_head"])))
        with patch("torch.load",return_value=bad):
            with self.assertRaises(RuntimeError):load_bundle(self.sam_path,"A0",allow_corner_candidate=True)
        with self.assertRaisesRegex(ValueError,"explicit return_corner"):
            self.model(self.rgb,self.K,self.cad,mask_aux=self.aux,identities=self.ids)
        with self.assertRaisesRegex(ValueError,"scheduled training refused"):
            validate_run_gate(self.cfg,{"launch_allowed":True,"mode":"integration"})
        drift=copy.deepcopy(self.cfg)
        drift["corner_counterpart"]["train_sam_cache"]["sha256"]="a"*64
        with self.assertRaisesRegex(ValueError,"configuration changed"):optimizer_for(self.model,drift)
        with self.assertRaisesRegex(ValueError,"configuration changed"):bundle_snapshot(self.model,self.parent,drift,0)
        bad=copy.deepcopy(self.cfg);bad["corner_protocol"]["quality_ready"]=True
        with self.assertRaises(ValueError):validate_corner_config(bad)
        bad=copy.deepcopy(self.cfg);bad["independent_head_dependencies"]=bad["independent_head_dependencies"][:-1]
        with self.assertRaises(ValueError):validate_corner_config(bad)

    def test_parent_hash_load_race_refused_for_SAM_and_corner_parent(self):
        from model.action4_release_bundle import _load_verified_checkpoint
        states=[self.sam_state,bundle_snapshot(self.model,self.parent,self.cfg,0)]
        for index,state in enumerate(states):
            original=OUT/f"parent_race_{index}.pt";replacement=OUT/f"parent_replacement_{index}.pt"
            torch.save(state,original)
            other=dict(state);other["model"]=dict(state["model"])
            key=next(k for k,v in other["model"].items() if v.is_floating_point())
            other["model"][key]=other["model"][key]+.125
            # A valid same-selected-step bundle; only actual weight bytes differ.
            torch.save(other,replacement);expected=sha(original)
            real=hashlib.file_digest
            def swap(stream,*args,**kwargs):
                value=real(stream,*args,**kwargs)
                if Path(stream.name).resolve()==original.resolve():replacement.replace(original)
                return value
            cfg=copy.deepcopy(self.cfg)
            cfg.update(parent_bundle=str(original),parent_bundle_sha256=expected,parent_best_step=state["selected_step"])
            with patch("hashlib.file_digest",side_effect=swap):
                with self.assertRaisesRegex(ValueError,"Parent replaced"):
                    construct(cfg,"cpu")
            # A second race after hash checks, inside deserialization, must also reject.
            torch.save(state,original);torch.save(other,replacement);expected=sha(original)
            real_load=torch.load
            def replace_during_load(stream,*args,**kwargs):
                value=real_load(stream,*args,**kwargs);replacement.replace(original);return value
            with patch("torch.load",side_effect=replace_during_load):
                with self.assertRaisesRegex(ValueError,"during weight load"):
                    _load_verified_checkpoint(original,expected)

    def test_existing_SAM_default_format_preserved(self):
        self.assertEqual(self.sam_state["schema"],SAM_SCHEMA)
        self.assertNotIn("corner_head",self.sam_state)
        head,base,state=load_bundle(self.sam_path,"A0")
        self.assertFalse(hasattr(head,"corner_head"))
        sam=JointSurfaceRGB(self.sam_path,enable_sam=True).eval()
        with torch.no_grad():
            default=sam(self.rgb,self.K,self.cad,mask_aux=self.aux,identities=self.ids)
            extra,aux=sam(self.rgb,self.K,self.cad,mask_aux=self.aux,identities=self.ids,return_aux=True)
        output_equal(self,default,extra)
        self.assertFalse(any(k.startswith("corner_head.") for k in sam.state_dict()))


if __name__=="__main__":unittest.main(verbosity=2)
