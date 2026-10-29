"""Opt-in actual training-input consumption; not accuracy or formal training."""
import copy,hashlib,json,os,sys,time,unittest
from pathlib import Path
import torch
from data.action4_paths import PROJECT_ROOT as R,contained_path

ENABLED=os.environ.get("ACTION4_REAL_SAM_CONSUMPTION")=="1"
OUT=R/"outputs/action4_sam_real_consumption_v001"
CACHE=R/"outputs/action4_sam_selectedB0_train_cache_v001"

def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()

def module_digest(module):
    value=hashlib.sha256()
    for name,tensor in module.state_dict().items():
        value.update(name.encode());value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()

def same_output(test,a,b):
    test.assertEqual(set(a),set(b))
    for key in a:
        if isinstance(a[key],dict):same_output(test,a[key],b[key])
        else:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

@unittest.skipUnless(ENABLED,"explicit actual train-cache integration opt-in required")
class RealSamConsumptionTests(unittest.TestCase):
    def test_actual_dataset_losses_updates_and_reload(self):
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"),"")
        self.assertTrue(set(os.sched_getaffinity(0))<=set(range(24,96)))
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261025)
        self.assertTrue((CACHE/"COMPLETE.json").is_file(),"Do not consume partial train cache")
        completed=json.loads((CACHE/"COMPLETE.json").read_text())
        self.assertEqual(completed["status"],"TRAIN_CACHE_AND_ASSOCIATION_PRODUCED_PENDING_ROOT_ACTUAL_STAGE2UNITS_REVIEW")
        self.assertEqual(completed["coverage"]["nonignored_gt_units"],10400)
        OUT.mkdir(parents=True,exist_ok=False)
        def write(name,value):
            (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False))
        from data.action4_stage2_data import Stage2Units
        from data.action4_sam_roi_routing import summarize_routes
        from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux,PROTOCOL_ID
        from model.action4_sam_joint import QUALITY_PREFIX,QUALITY_KEY,stack_sam_items
        from model.training.action4_stage2_core import construct,optimizer_for,loss_sums_on_batch,bundle_snapshot
        from model.training.action4_supervision import local_counts,weighted_objective,step_if_supervised,A0_KEYS
        from model.action4_joint_surface_rgb import JointSurfaceRGB
        from model.action4_release_bundle import load_bundle
        initial_sources={}
        for module in tuple(sys.modules.values()):
            path=getattr(module,"__file__",None)
            if path:
                source_path=Path(path).resolve()
                if source_path.is_relative_to(R/"src"):
                    initial_sources[str(source_path.relative_to(R))]=digest(source_path)
        baseline=json.loads((R/"runtime/action4_runs/p2_a0_parallel80_eval10_v001/source/resolved_config.json").read_text())
        cache=json.loads((CACHE/"train/CACHE_SPEC.json").read_text())
        parent=R/"models/default/A0.pt"
        registry=json.loads((R/"configs/model/action4_best_registry_v003.json").read_text())["entries"]["A0"]
        self.assertEqual(digest(parent),registry["sha256"])
        cfg={k:copy.deepcopy(baseline[k]) for k in ("dataset_root","validation_receipt","surface_source","roi_jitter","seed","loss_weights")}
        cfg.update(role="A0",mode="integration",train_backbone=False,parent_bundle=str(parent.relative_to(R)),
            parent_best_step=registry["selected_step"],parent_bundle_sha256=registry["sha256"],
            a0_variant="a0_sam_image",sam_enabled=True,sam_protocol_id=PROTOCOL_ID,
            surface_lr=1e-5,lr=5e-5,weight_decay=1e-4,sam_adapter_lr=1e-4,sam_quality_lr=1e-4,
            photometric=False,train_indices=[0,1,100,101,5198,5199,10398,10399],
            sam_cache=cache,sam_training_associations=completed["association_binding"],
            independent_head_dependencies=[{"parameter_prefixes":[QUALITY_PREFIX],"losses":[QUALITY_KEY]},
                {"parameter_prefixes":["head.surface"],"losses":list(A0_KEYS)},
                {"parameter_prefixes":["head.pose"],"losses":[k for k in A0_KEYS if k not in ("xyz","region","visible_mask","amodal_mask")]}])
        cfg["loss_weights"][QUALITY_KEY]=.1
        for key,relative in (("sam_mask_augmentation","configs/data/action4_sam_mask_augmentation_v001.json"),
                             ("sam_roi_routing","configs/data/action4_sam_roi_routing_v001.json")):
            cfg[key]={"path":relative,"sha256":digest(R/relative)}
        write("CONFIG.json",cfg);write("STATUS.json",{"status":"VERIFYING_REAL_TRAIN_CACHE"})
        started=time.monotonic()
        data=Stage2Units(cfg,sam_factories=(FeatureIdentity,MaskAux))
        initialization_seconds=time.monotonic()-started
        self.assertEqual(len(data.data),10400);self.assertEqual(len(data),8)
        items=[data[(i,0,True,8)] for i in range(8)]
        details=[]
        for item in items:
            self.assertTrue(torch.isfinite(item["roi"]).all())
            self.assertEqual(tuple(item["roi"].shape),(3,448,448))
            self.assertEqual(item["metadata"]["sample_id"],item["identity"].image_id)
            self.assertEqual(item["metadata"]["corner_gt224"]["image_id"],item["identity"].image_id)
            self.assertEqual(str(item["metadata"]["instance_id"]),str(item["sam_input"]["training_gt_instance_id"]))
            details.append({"data_index":item["data_index"],"identity":list(item["identity"].__dict__.values()),
                "GT_instance_id":str(item["metadata"]["instance_id"]),
                "sam_input":item["sam_input"],"active_visible_mask_pixels":int(item["mask_aux"].visible.bool().sum()),
                "mask_available":bool(item["mask_aux"].available.flatten()[0]),
                "dropout":bool(item["mask_aux"].dropout[0]),
                "loss_counts":local_counts([item],sam_enabled=True)})
        write("ACTUAL_INPUTS.json",{"scope":"Eight predeclared actual train units; no eval/test training",
            "provider_initialization_seconds":initialization_seconds,"units":details,
            "routes":summarize_routes([x["metadata"]["sam_roi_routing"] for x in items])})
        write("STATUS.json",{"status":"ACTUAL_TWO_CPU_UPDATES"})
        model,parent_state=construct(cfg,"cpu");model.train()
        frozen=module_digest(model.base)
        optimizer=optimizer_for(model,cfg);scheduler=torch.optim.lr_scheduler.StepLR(optimizer,100)
        before=model.sam_adapter.projection.weight.detach().clone()
        batch=items[:2];counts=local_counts(batch,sam_enabled=True)
        # Fixed first pair, not chosen by quality or gradient outcomes.
        self.assertTrue(any(x["mask_aux"].available.bool().any() and x["mask_aux"].visible.bool().any() for x in batch))
        updates=[]
        for step in (1,2):
            optimizer.zero_grad(set_to_none=True)
            sums=loss_sums_on_batch(model,batch,cfg,"cpu",epoch_one_based=1)
            self.assertEqual(set(sums),set(counts))
            self.assertTrue(all(torch.isfinite(x) for x in sums.values()))
            objective=weighted_objective(sums,counts,cfg["loss_weights"],1)
            objective.backward()
            gradient=sum(float(p.grad.abs().sum()) for p in model.sam_adapter.projection.parameters() if p.grad is not None)
            self.assertGreater(gradient,0)
            did,norm=step_if_supervised(model,optimizer,scheduler,counts,cfg["loss_weights"],cfg["independent_head_dependencies"])
            self.assertTrue(did)
            updates.append({"optimizer_step":step,"objective":float(objective.detach()),"projection_gradient_l1":gradient,"gradient_norm":norm,"eligible_counts":counts})
        self.assertFalse(torch.equal(before,model.sam_adapter.projection.weight))
        self.assertEqual(module_digest(model.base),frozen)
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))
        model.eval();rgb=torch.stack([x["roi"] for x in batch]);K=torch.stack([x["geometry"]["K"] for x in batch]);cad=torch.stack([x["geometry"]["canonical"] for x in batch])
        aux,ids,_,_=stack_sam_items(batch,"cpu",rgb.dtype)
        with torch.no_grad():
            expected=model(rgb,K,cad,mask_aux=aux,identities=ids)
            missing=model(rgb,K,cad,mask_aux=None,identities=ids)
        mask_effect=float((expected["corners"]-missing["corners"]).abs().max())
        self.assertGreater(mask_effect,0)
        state=bundle_snapshot(model,parent_state,cfg,2);saved=OUT/"ENGINEERING_TWO_UPDATES.pt";torch.save(state,saved)
        _,_,loaded=load_bundle(saved,"A0","cpu")
        self.assertEqual(loaded["selected_step"],2)
        restored=JointSurfaceRGB(saved,enable_sam=True,parent_bundle_sha256=digest(saved)).eval()
        with torch.no_grad():actual=restored(rgb,K,cad,mask_aux=aux,identities=ids)
        same_output(self,expected,actual)
        self.assertEqual(digest(parent),registry["sha256"])
        source={}
        for module in tuple(sys.modules.values()):
            path=getattr(module,"__file__",None)
            if path:
                p=Path(path).resolve()
                if p.is_relative_to(R/"src"):source[str(p.relative_to(R))]=digest(p)
        for relative,expected_sha in initial_sources.items():
            self.assertEqual(digest(R/relative),expected_sha,"Source changed during integration: "+relative)
        write("COMPLETE.json",{"status":"ACTUAL_TRAIN_INPUT_TWO_UPDATES_RELOAD_PASS_NOT_EFFECT_ACCEPTANCE",
            "trained_units":[x["data_index"] for x in batch],"inspected_units":cfg["train_indices"],"updates":updates,
            "mask_vs_missing_corners_max_abs112":mask_effect,"frozen_backbone_parameters_and_buffers_unchanged":True,
            "strict_reload_exact":True,"default_unchanged":True,"formal_training_started":False,"GPU_used":False,
            "source_sha256":source,"data_manifest_sha256":cfg["surface_source"]["manifest_sha256"]["train"],
            "train_cache_complete_receipt":str((CACHE/"COMPLETE.json").relative_to(R)),
            "scope":"CPU integration with existing parent, fresh SAM adapter and actual train inputs; not selected stage3 model or full650 accuracy"})
        write("STATUS.json",{"status":"COMPLETE","formal_training_started":False})

if __name__=="__main__":unittest.main(verbosity=2)
