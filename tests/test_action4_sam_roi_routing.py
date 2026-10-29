
"""Small fabricated native-format fixtures test contracts, never claim SAM inference."""
import copy,json,unittest,uuid,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R
from data.action4_sam_native_export import export,sha,canonical,verify_export_receipt
from data.action4_sam_roi_inputs import native_key,NativeMaskIdentity
from data.action4_sam_cache_provider import SamCacheProvider


import torch
from unittest.mock import patch
from data.action4_sam_roi_routing import POLICY,decision,SamRoiRouter,summarize_routes,validate_config
from data.action4_training_crop_specs import TrainingCropSpec

class RoiRoutingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.split=getattr(self,"split","train")
        self.base=R/'outputs/action4_sam_roi_routing_preparation_v001'/uuid.uuid4().hex
        self.base.mkdir(parents=True);self.native=self.base/'native';self.native.mkdir()
        def jsonfile(path,value,readonly=False):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            if readonly:path.chmod(0o444)
            return {'path':str(path.relative_to(R)),'sha256':sha(path)}
        self.jsonfile=jsonfile
        bundle=self.base/'B0.pt';bundle.write_bytes(b'CPU fixture, not a trained model');bundle.chmod(0o444)
        self.bundle=bundle
        run=jsonfile(self.base/'RUN_SPEC.json',{'schema':'action4.immutable_run.v1','code_root':'fixture/source',
            'mutable_worktree_not_used_by_children':True,'project_code_sha256':'b'*64},True)
        model={'path':str(bundle.relative_to(R)),'sha256':sha(bundle)}
        seal=jsonfile(self.base/'SEALED_ROLE.json',{'schema':'action4.sealed_independent_role.v2','role':'B0','strict_load':True,
            'selected_step':9500,'config_id':'fixture','bundle':model['path'],'sha256':model['sha256'],
            'run_spec':run['path'],'run_spec_sha256':run['sha256'],'actual_code_root':'fixture/source',
            'completion':{'status':'COMPLETION_METADATA_READY','role':'B0','best_step':9500,'config_id':'fixture','evaluations':653}},True)
        raw=[{'grid_index':j,'score':.8 if j<2 else .2,'bbox_full_xyxy':[1.,1.,4.,5.]} for j in range(3)]
        preds=jsonfile(self.base/'PREDICTIONS.json',[{'sample_id':'fixture','instances':raw}])
        source={'sealed_role':seal,'run_spec':run,'b0_bundle':model,'selected_step':9500,'config_id':'fixture','predictions':preds}
        rgb=np.arange(8*10*3,dtype=np.uint8).reshape(8,10,3);image=self.base/'image.png';Image.fromarray(rgb).save(image);self.rgb=rgb
        cs=[]
        for j,r in enumerate(raw):
            cs.append({'candidate_id':f'b0:fixture:grid:{j}','identity_binding':{'sample_id':'fixture','grid_index':j,
                'raw_saved_ordinal':j,'prediction_file_sha256':preds['sha256']},'raw_saved_prediction':r,
                'source':'frozen_completed_B0_saved_prediction','score':r['score'],'published':j<2,
                'eligible_for_SAM_diagnostic_prompt':j<2,'bbox_full_native_xyxy':r['bbox_full_xyxy'],
                'bbox_sam_native_xyxy':r['bbox_full_xyxy'],'bbox_sam_normalized_xywh':[.1,.125,.3,.5]})
        row={'schema':'action4.sam.prediction_prompt_image.v2','image_id':'fixture','image_path':str(image.relative_to(R)),
            'image_sha256':sha(image),'native_shape_hw':[8,10],'pixel_space_id':'native.fixture','pixel_convention':'integer_centers',
            'split':self.split,'rgb_variant':'native_original','source_receipt':source,'raw_saved_candidate_count':3,'SAM_prompt_count':2,
            'GT_used_for_prompt_selection':False,'candidates':cs}
        self.row=row;src=self.base/'source';src.mkdir()
        prompt=src/'prompts.jsonl';prompt.write_text(json.dumps(row)+'\n')
        contract=jsonfile(src/'input.json',{'prediction_only':True})
        code=jsonfile(src/'src/eval/action4_sam_image_eval.py',{'fixture':True},True)
        asset=jsonfile(src/'asset.json',{'sha256':'a'*64})
        official=jsonfile(src/'upstream/sam3/sam3/test.py',{'fixture_official':True},True)
        sm={'official_commit':'fixture-commit','official_source_relative':'upstream/sam3','schema':'action4.sam.immutable_source.v1','files_sha256':{'src/eval/action4_sam_image_eval.py':code['sha256'],'asset.json':asset['sha256'],'upstream/sam3/sam3/test.py':official['sha256']}}
        sm['identity']=canonical(sm);jsonfile(src/'SNAPSHOT_MANIFEST.json',sm)
        cfg={'schema':'action4.sam.single_image_native.v2','upstream_role':'frozen_completed_B0_post_soft_nms',
            'split':self.split,'rgb_variant':'native_original','source_contract':source,'expected_images':1,'expected_saved_candidates':3,
            'expected_prompts':2,'manifest':str(prompt.relative_to(R)),'manifest_sha256':sha(prompt),'input_contract':contract['path'],
            'source_snapshot':str(src.relative_to(R)),'output':str(self.native.relative_to(R)),'native':{'text':'camera','asset_receipt':asset['path'],'source_commit':'fixture-commit'}}
        config_binding=jsonfile(src/'config.json',cfg,True)
        sm.pop('identity');sm['config_file']='config.json';sm['files_sha256']['config.json']=config_binding['sha256']
        sm['identity']=canonical(sm);jsonfile(src/'SNAPSHOT_MANIFEST.json',sm)
        lock={'schema':cfg['schema'],'config':cfg,'config_identity':canonical(cfg),'code_sha256':{code['path']:code['sha256']},
            'manifest_sha256':sha(prompt),'input_contract_sha256':contract['sha256'],'source_snapshot_identity':sm['identity']}
        lock['identity']=canonical(lock);self.lock=lock;jsonfile(self.native/'RUN_LOCK.json',lock)
        self.progress={'status':'NATIVE_TRANSPORT_COMPLETE_MASK_QUALITY_NOT_SCORED','lock_identity':lock['identity'],
            'backend_kind':'sam31_official_native','completed_jobs':2,'expected_prompt_jobs':2,'image_domain_count':1,
            'upstream_saved_candidates':3,'upstream_unprompted_candidates':1}
        jsonfile(self.native/'PROGRESS.json',self.progress)
        jsonfile(self.native/'NATIVE_LOAD.json',{'status':'LOADED_ZERO_FINAL_MISSING_UNEXPECTED','source_snapshot_identity':sm['identity'],
            'source_commit':'fixture-commit','weight_sha256_from_prior_validation':'a'*64})
        self.mask=np.zeros((8,10),bool);self.mask[1:5,1:4]=True
        for j,c in enumerate(cs[:2]):
            key=canonical({'lock':lock['identity'],'image_id':'fixture','image_sha256':row['image_sha256'],'candidate':c,'text':'camera'})
            job=self.native/'jobs'/key;attempt=job/'attempt_01';attempt.mkdir(parents=True)
            rp=attempt/'RESULT.json';rawpath=attempt/'raw.npz';np.savez(rawpath,raw__outputs__out_binary_masks=self.mask[None],raw__outputs__out_obj_ids=np.array([7],dtype=np.int64))
            regbinding={'official_commit':'fixture-commit','session_id':'session'+str(j),'obj_id':7,'generation':0,'track_id':c['candidate_id'],
                'pixel_identity':row['pixel_space_id'],'dependency_identity':lock['identity']}
            pix=hashlib.sha256(json.dumps({'dtype':'bool','shape':[8,10]},separators=(',',':')).encode()+b'\0'+self.mask.tobytes()).hexdigest()
            reg={'pixel_sha256':pix,'binding':regbinding,'binding_identity':canonical(regbinding)}
            reg['mask_identity']='sam-mask-sha256:'+canonical({'binding':reg['binding_identity'],'pixels':pix})
            result={'schema':cfg['schema'],'task_key':key,'lock_identity':lock['identity'],'image_id':'fixture',
                'image_sha256':row['image_sha256'],'candidate_id':c['candidate_id'],'prediction_id':c['candidate_id'],
                'identity_binding':c['identity_binding'],'pixel_space_id':row['pixel_space_id'],'source_receipt':source,
                'split':self.split,'rgb_variant':'native_original','close_success':True,'response_valid':True,
                'execution_status':'COMPLETED','backend_kind':'sam31_official_native','available':j==0,
                'status':'target_present' if j==0 else 'ambiguous_identity','binding_status':'unique' if j==0 else 'ambiguous_identity',
                'session_id':'session'+str(j),'bound_obj_id':7,'generation':0,'mask_registration':reg,
                'native_candidates':[{'obj_id':7,'raw_index':0,'mask_pixels':12,'target_mask_coverage':1.,
                    'mask_storage':{'npz':str(rawpath.relative_to(R)),'raw_sha256':sha(rawpath),'array_key':'raw__outputs__out_binary_masks','row_index':0}}]}
            jsonfile(rp,result)
            jsonfile(job/'COMPLETED.json',{'task_key':key,'lock_identity':lock['identity'],'result_path':str(rp.relative_to(R)),
                'files_sha256':{str(rp.relative_to(R)):sha(rp),str(rawpath.relative_to(R)):sha(rawpath)}})
    def do_export(self):
        out=self.base/'export';export(self.native,out)
        spec=json.loads((out/'CACHE_SPEC.json').read_text());index=json.loads((out/'INDEX.json').read_text())
        return out,spec,index

    def configured(self):
        from data.action4_sam_training_associations import build_associations
        from data.action4_sam_cache_provider import SamCacheProvider
        from data import DataAdapter
        out,spec,index=self.do_export()
        cad=np.array([[x,y,z] for z in (-.01,.01) for y in (-.04,.04) for x in (-.03,.03)])
        K=np.array([[50.,0,2.5],[0,50.,3.],[0,0,1.]])
        T=np.eye(4);T[2,3]=1.
        cam=cad+T[:3,3];h=cam@K.T;corners=h[:,:2]/h[:,2:]
        visible=self.base/"gt_visible.png";amodal=self.base/"gt_amodal.png"
        Image.fromarray((self.mask*255).astype(np.uint8)).save(visible);Image.fromarray((self.mask*255).astype(np.uint8)).save(amodal)
        obj={"instance_id":1,"bbox_full_xyxy":[1.,1.,4.,5.],"ignore":False,
             "canonical8_geometry":{"object_to_capture_cv_m":T.tolist()},"corners_px":corners.tolist(),
             "corner_visible":[True]*4+[False]*4,"occlusion_inframe":.5,
             "mask_visible":str(visible.relative_to(R)),"mask_amodal_expanded":str(amodal.relative_to(R)),
             "amodal_expanded_origin_px":[0,0],"amodal_expanded_shape_hw":[8,10]}
        record={"sample_id":"fixture","split":"train","status":"training_accepted","image":self.row["image_path"],
            "rgb_sha256":self.row["image_sha256"],"width":10,"height":8,"camera":{"K":K.tolist()},
            "corners_3d_mm":(cad*1000).tolist(),"instances":[obj]}
        ann=self.base/"annotations.jsonl";ann.write_text(json.dumps(record)+"\n")
        ap=R/"configs/data/action4_sam_training_association_v001.json"
        association=build_associations(spec,{"path":str(ann.relative_to(R)),"sha256":sha(ann)},
            {"path":str(ap.relative_to(R)),"sha256":sha(ap)},self.base/"ASSOCIATION.json")
        assoc=json.loads((R/association["path"]).read_text())
        rp=R/"configs/data/action4_sam_roi_routing_v001.json"
        route_binding={"path":str(rp.relative_to(R)),"sha256":sha(rp)}
        provider=SamCacheProvider(spec,split="train",training_associations=association,roi_routing=route_binding)
        item={"record":record,"instance":0,"image":torch.from_numpy(self.rgb.copy()).permute(2,0,1).contiguous()}
        points=np.array([[-.02,-.02,0],[.02,-.02,0],[-.02,.02,0],[.02,.02,0]])
        camera=points+T[:3,3];hom=camera@K.T;pixels=hom[:,:2]/hom[:,2:]
        labels=self.base/"labels";labels.mkdir()
        np.savez(labels/"fixture_i1.npz",xyz_object_m=points,native_pixels=pixels,K=K,object_to_capture=T)
        review=self.jsonfile(self.base/"surface_review.json",{"candidate_training_allowed":True})
        regions=self.jsonfile(self.base/"regions.json",{"xyz_object_m":np.tile(points,(16,1)).tolist()})
        surface={"admission":review["path"],"regions":regions["path"],"label_roots":{"train":str(labels.relative_to(R))}}
        cfg={"role":"A0","mode":"integration","a0_variant":"a0_sam_image","sam_enabled":True,"seed":27,
            "sam_cache":spec,"sam_training_associations":association,"sam_roi_routing":route_binding,
            "roi_jitter":copy.deepcopy(POLICY["gt_jitter"]),"validation_receipt":"fixture_unused","surface_source":surface}
        return provider,item,cfg,assoc,index
    def test_proportions_reorder_and_routefirst_fallback(self):
        provider,item,cfg,assoc,index=self.configured()
        choices=[decision(i,0,"fixture","1")["requested_route"] for i in range(20000)]
        counts={r:choices.count(r) for r in set(choices)}
        self.assertLess(abs(counts["gt_jitter"]/20000-.5),.015)
        self.assertLess(abs(counts["extra_offset"]/20000-.2),.012)
        self.assertLess(abs(counts["prediction_full"]/20000-.15),.012)
        self.assertLess(abs(counts["prediction_coarse"]/20000-.15),.012)
        seed=next(i for i,r in enumerate(choices) if r=="prediction_coarse")
        crop,trace=provider.training_crop_spec(item,seed=seed,epoch=0)
        self.assertEqual(crop.requested_route,"prediction_coarse");self.assertEqual(crop.route,"prediction_full")
        self.assertEqual(trace["fallback_reason"],"requested_route_unavailable")
        self.assertFalse(trace["coarse_policy_ready"]);self.assertEqual(crop.prediction_id,"b0:fixture:grid:0")
        again,_=provider.training_crop_spec(item,seed=seed,epoch=0);self.assertEqual(crop,again)
        # Dataset/source-index order is not an input to route seed identity.
        self.assertEqual(decision(seed,0,"fixture","1"),decision(seed,0,"fixture","1"))
        missing=copy.deepcopy(assoc);missing["rows"][0].update(status="missing",prediction_id=None,qualified_prediction_ids=[])
        router=SamRoiRouter(cfg["sam_roi_routing"],missing,{r["image_id"]:r for r in index["images"]})
        absent,why=router.choose(item,seed=seed,epoch=0)
        self.assertEqual(absent.route,"gt_jitter");self.assertEqual(why["fallback_reason"],"predicted_roi_missing")
        self.assertIsNone(absent.prediction_id)
    def test_prediction_ranking_close_scores_ids_duplicates(self):
        provider,item,cfg,assoc,index=self.configured()
        raw=json.loads((R/assoc["predictions"]["path"]).read_text())
        raw[0]["instances"][0]["score"]=.80000001;raw[0]["instances"][1]["score"]=.80000002
        binding=self.jsonfile(self.base/"ranking.json",raw)
        a=copy.deepcopy(assoc);a["predictions"]=binding;a["source_contract"]["predictions"]=binding
        router=SamRoiRouter(cfg["sam_roi_routing"],a,{r["image_id"]:r for r in index["images"]})
        seed=next(i for i in range(100) if decision(i,0,"fixture","1")["requested_route"]=="prediction_full")
        selected,trace=router.choose(item,seed=seed,epoch=0)
        self.assertEqual(selected.prediction_id,"b0:fixture:grid:1")
        self.assertEqual(trace["fixed_SAM_prediction_id"],"b0:fixture:grid:0")
        raw[0]["instances"].reverse();binding=self.jsonfile(self.base/"reordered.json",raw)
        a["predictions"]=binding;a["source_contract"]["predictions"]=binding
        reversed_router=SamRoiRouter(cfg["sam_roi_routing"],a,{r["image_id"]:r for r in index["images"]})
        self.assertEqual(reversed_router.choose(item,seed=seed,epoch=0)[0],selected)
        bad=copy.deepcopy(a);bad["rows"][0]["qualified_prediction_ids"]=["b0:another:grid:1"]
        with self.assertRaisesRegex(ValueError,"wrong image"):SamRoiRouter(cfg["sam_roi_routing"],bad,{r["image_id"]:r for r in index["images"]})
        raw[0]["instances"][1]=copy.deepcopy(raw[0]["instances"][0])
        binding=self.jsonfile(self.base/"duplicate.json",raw);a["predictions"]=binding;a["source_contract"]["predictions"]=binding
        with self.assertRaisesRegex(ValueError,"duplicate"):SamRoiRouter(cfg["sam_roi_routing"],a,{r["image_id"]:r for r in index["images"]})
    def test_actual_stage2_crop_specs_consumption_and_empty_support(self):
        from data.action4_stage2_data import Stage2Units
        from data.action4_online_surface_targets import prepare_online_surface_batch
        from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux
        from model.training.action4_supervision import local_counts
        provider,item,cfg,_,_=self.configured()
        class Dataset:
            records=[item["record"]];units=[(0,0)]
            def __len__(self):return 1
            def __getitem__(self,index):return item
        with patch("data.action4_stage2_data.ComponentData",return_value=Dataset()):
            units=Stage2Units(cfg,sam_factories=(FeatureIdentity,MaskAux))
        traces=[]
        for route in ("gt_jitter","prediction_full","prediction_coarse","extra_offset"):
            epoch=next(e for e in range(200) if decision(cfg["seed"],e,"fixture","1")["requested_route"]==route)
            crop,trace=units.sam_provider.training_crop_spec(item,seed=cfg["seed"],epoch=epoch)
            unit=units[(0,epoch,True,1)]
            rgb,targets,geometry,meta=prepare_online_surface_batch([item],"train",crop_specs=[crop],surface_source=cfg["surface_source"])
            torch.testing.assert_close(unit["roi"],rgb[0],rtol=0,atol=0)
            torch.testing.assert_close(unit["surface_masks"],targets["masks"][0],rtol=0,atol=0)
            for key in geometry:torch.testing.assert_close(unit["geometry"][key],geometry[key][0],rtol=0,atol=0)
            self.assertEqual(unit["metadata"]["affine_hm"],meta[0]["affine_hm"])
            self.assertEqual(unit["metadata"]["sam_roi_routing"]["fixed_SAM_prediction_id"],"b0:fixture:grid:0")
            self.assertEqual(unit["sam_input"]["prediction_id"],"b0:fixture:grid:0")
            self.assertEqual(unit["augmentation_seed"],crop.augmentation_seed)
            traces.append(unit["metadata"]["sam_roi_routing"])
        self.assertEqual(len(units),1)
        self.assertEqual(summarize_routes(traces)["active_units"],4)
        dummy=units[(0,0,False,1)];self.assertEqual(summarize_routes([dummy["metadata"]["sam_roi_routing"]])["active_units"],0)
        # Transport/qualification counterexample: a crop with zero visible support stays a unit.
        far=TrainingCropSpec("fixture","1",(100.,100.,104.,104.),"extra_offset","extra_offset",1,
            cfg["sam_roi_routing"]["sha256"],"engineering_empty_support")
        rgb,targets,g,meta=prepare_online_surface_batch([item],"train",crop_specs=[far],surface_source=cfg["surface_source"])
        self.assertEqual(meta[0]["target_visible_input_support_count"],0)
        plain={"active":True,"metadata":meta[0],"surface_masks":targets["masks"][0],"surface_sparse":targets["sparse"][0],
               "geometry":{k:v[0] for k,v in g.items()}}
        counts=local_counts([plain]);self.assertEqual(counts["visible_corners"],0);self.assertEqual(counts["hidden_corners"],0)
    def test_returned_trace_deep_mutation_cannot_change_router(self):
        provider,item,cfg,assoc,index=self.configured()
        router=provider.roi_router
        seed=next(i for i in range(100) if decision(i,0,"fixture","1")["requested_route"]=="prediction_full")
        before={k:copy.deepcopy(getattr(router,k)) for k in ("predictions","policy","policy_binding","prediction_binding","source_contract","associations")}
        original,trace=provider.training_crop_spec(item,seed=seed,epoch=0)
        saved_trace=copy.deepcopy(trace)
        trace["selected_crop_prediction"]["bbox_full_xyxy"][0]+=100
        trace["selected_crop_prediction"]["score"]=-100
        trace["policy"]["gt_jitter"]["scale_range"][0]=.001
        trace["policy"]["fallback"].append("invented")
        trace["policy_binding"]["sha256"]="0"*64
        trace["prediction_source"]["path"]="outside-owned-source"
        trace["source_contract"]["b0_bundle"]["sha256"]="0"*64
        trace["source_contract"]["predictions"]["path"]="wrong-predictions"
        again,repeated=provider.training_crop_spec(item,seed=seed,epoch=0)
        self.assertEqual(original,again);self.assertEqual(saved_trace,repeated)
        for key,value in before.items():self.assertEqual(getattr(router,key),value,key)
        self.assertIsNot(repeated["selected_crop_prediction"],router.predictions["fixture"][original.prediction_id])
    def test_fixed_counter_domain_and_inactive_exclusion(self):
        from data.action4_sam_roi_routing import ROUTE_COUNT_ROUTES,ROUTE_COUNT_FALLBACKS,ROUTE_COUNT_FIELDS
        empty=summarize_routes([])
        self.assertEqual(len(ROUTE_COUNT_FIELDS),11)
        self.assertEqual(tuple(empty["requested"]),ROUTE_COUNT_ROUTES)
        self.assertEqual(tuple(empty["effective"]),ROUTE_COUNT_ROUTES)
        self.assertEqual(tuple(empty["fallbacks"]),ROUTE_COUNT_FALLBACKS)
        self.assertTrue(all(v==0 for group in ("requested","effective","fallbacks") for v in empty[group].values()))
        trace={"active":True,"requested_route":"prediction_coarse","effective_route":"prediction_full","fallback_reason":"requested_route_unavailable"}
        summary=summarize_routes([trace,dict(trace,active=False)])
        self.assertEqual(summary["active_units"],1);self.assertEqual(summary["effective"]["prediction_full"],1)
        self.assertEqual(summary["effective"]["prediction_coarse"],0)
        with self.assertRaises(ValueError):summarize_routes([dict(trace,effective_route="unknown")])

    def test_policy_ranges_early_gate_and_source_change(self):
        provider,item,cfg,_,_=self.configured()
        original=copy.deepcopy(cfg["roi_jitter"])
        for seed in range(200):
            crop,trace=provider.training_crop_spec(item,seed=seed,epoch=0)
            if trace["crop_parameters"]:
                settings=POLICY["extra_offset"] if crop.route=="extra_offset" else POLICY["gt_jitter"]
                p=trace["crop_parameters"]
                self.assertTrue(all(abs(x)<=settings["center_fraction"] for x in p["center_fraction_xy"]))
                self.assertLessEqual(settings["scale_range"][0],p["scale"]);self.assertLessEqual(p["scale"],settings["scale_range"][1])
        self.assertEqual(cfg["roi_jitter"],original)
        for change in ({"mode":"full_training"},{"role":"B0"},{"a0_variant":"a0_baseline"},{"corner_enabled":True},{"roi_jitter":{"center_fraction":.2}}):
            bad=dict(cfg,**change)
            with self.assertRaises(ValueError):validate_config(bad,"train")
        with self.assertRaises(ValueError):validate_config(cfg,"eval")
        path=R/provider.roi_router.prediction_binding["path"];path.write_text("[]")
        with self.assertRaisesRegex(ValueError,"predictions changed"):provider.training_crop_spec(item,seed=1,epoch=0)

if __name__=="__main__":unittest.main()
