
"""Small fabricated native-format fixtures test contracts, never claim SAM inference."""
import copy,json,unittest,uuid,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R
from data.action4_sam_native_export import export,sha,canonical,verify_export_receipt
from data.action4_sam_roi_inputs import native_key,NativeMaskIdentity
from data.action4_sam_cache_provider import SamCacheProvider

from data.action4_sam_training_associations import build_associations,derive_rows,ASSIGNMENT

class TrainingAssociationTests(unittest.TestCase):
    def setUp(self):
        self.split=getattr(self,"split","train")
        self.base=R/'outputs/action4_sam_training_association_preparation_v001'/uuid.uuid4().hex
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

    def annotation(self,instances=None,**extra):
        instances=instances if instances is not None else [{"instance_id":1,"bbox_full_xyxy":[1.,1.,4.,5.],"ignore":False}]
        record={"sample_id":"fixture","split":"train","status":"training_accepted","image":self.row["image_path"],
            "rgb_sha256":self.row["image_sha256"],"instances":instances,**extra}
        path=self.base/("annotations_"+uuid.uuid4().hex+".jsonl");path.write_text(json.dumps(record)+"\n")
        return record,{"path":str(path.relative_to(R)),"sha256":sha(path)}
    def build(self,instances=None,**extra):
        out,spec,index=self.do_export();record,ann=self.annotation(instances,**extra)
        policy_path=R/"configs/data/action4_sam_training_association_v001.json"
        policy={"path":str(policy_path.relative_to(R)),"sha256":sha(policy_path)}
        binding=build_associations(spec,ann,policy,self.base/"ASSOCIATIONS.json")
        return record,spec,index,binding,json.loads((R/binding["path"]).read_text())
    def test_production_full_coverage_and_selection(self):
        record,spec,index,binding,assoc=self.build()
        self.assertEqual(assoc["coverage"]["nonignored_gt_units"],1)
        self.assertEqual(assoc["rows"][0]["prediction_id"],"b0:fixture:grid:0")
        provider=SamCacheProvider(spec,split="train",training_associations=binding)
        provider.validate_image_coverage([record])
        self.assertTrue(provider.production_associations)
        self.assertEqual(provider.associations[("fixture","1")]["native_status"],"available")
    def test_ignore_missing_unknown_and_no_unit_drop(self):
        units=[{"instance_id":1,"bbox_full_xyxy":[6.,1.,9.,5.]},
               {"instance_id":2,"ignore":True,"bbox_full_xyxy":[1.,1.,4.,5.]},
               {"instance_id":3}]
        _,_,_,_,assoc=self.build(units)
        self.assertEqual(assoc["coverage"]["annotation_instance_count"],3)
        self.assertEqual(assoc["coverage"]["ignored_instance_count"],1)
        self.assertEqual(assoc["coverage"]["nonignored_gt_units"],2)
        self.assertEqual([r["status"] for r in assoc["rows"]],["missing","unknown"])
    def test_near_overlap_multiple_gt_is_ambiguous(self):
        units=[{"instance_id":1,"bbox_full_xyxy":[1.,1.,4.,5.]},
               {"instance_id":2,"bbox_full_xyxy":[1.2,1.,4.2,5.]}]
        _,_,_,_,assoc=self.build(units)
        self.assertTrue(all(r["status"]=="ambiguous" and r["prediction_id"] is None for r in assoc["rows"]))
    def test_duplicate_gt_and_resealed_omitted_or_wrong_image_rows_reject(self):
        record,spec,index,binding,assoc=self.build()
        for kind in ("omitted","wrong_image","wrong_id"):
            bad=copy.deepcopy(assoc)
            if kind=="omitted":bad["rows"]=[]
            elif kind=="wrong_image":bad["rows"][0]["image_id"]="other"
            else:bad["rows"][0]["prediction_id"]="b0:fixture:grid:999"
            bad["identity"]=canonical({k:v for k,v in bad.items() if k!="identity"})
            b=self.jsonfile(self.base/(kind+".json"),bad)
            with self.assertRaises(ValueError):SamCacheProvider(spec,split="train",training_associations=b)
        record["instances"].append(copy.deepcopy(record["instances"][0]))
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            derive_rows([record],{"fixture":[]},{"fixture":{"predictions":[]}}, {})
    def test_annotation_change_and_runtime_record_change_failclosed(self):
        record,spec,_,binding,assoc=self.build()
        provider=SamCacheProvider(spec,split="train",training_associations=binding)
        changed=copy.deepcopy(record);changed["instances"][0]["bbox_full_xyxy"]=[0.,0.,2.,2.]
        with self.assertRaisesRegex(ValueError,"Training record"):
            provider.validate_image_coverage([changed])
        path=R/assoc["annotations"]["path"];path.write_text(json.dumps(changed)+"\n")
        with self.assertRaises(ValueError):provider.validate_image_coverage([record])
        with self.assertRaises(ValueError):SamCacheProvider(spec,split="train",training_associations=binding)
    def test_native_unavailable_does_not_change_bbox_owner(self):
        for marker in (self.native/"jobs").glob("*/COMPLETED.json"):
            done=json.loads(marker.read_text());path=R/done["result_path"];result=json.loads(path.read_text())
            if result["available"]:
                result.update(available=False,status="ambiguous_identity",binding_status="ambiguous_identity")
                self.jsonfile(path,result);done["files_sha256"][done["result_path"]]=sha(path);self.jsonfile(marker,done)
        K=[[100.,0.,5.],[0.,100.,4.],[0.,0.,1.]]
        gt=self.base/"loss_only_gt.png";Image.fromarray((self.mask*255).astype(np.uint8)).save(gt)
        obj={"instance_id":1,"bbox_full_xyxy":[1.,1.,4.,5.],"mask_visible":str(gt.relative_to(R))}
        record,spec,_,binding,assoc=self.build([obj],camera={"K":K})
        self.assertEqual(assoc["rows"][0]["status"],"matched")
        self.assertEqual(assoc["rows"][0]["prediction_id"],"b0:fixture:grid:0")
        self.assertEqual(assoc["rows"][0]["native_status"],"ambiguous")
        provider=SamCacheProvider(spec,split="train",training_associations=binding)
        frame,predictions=provider.frame_predictions("fixture",self.rgb)
        self.assertIsNone(predictions[native_key(NativeMaskIdentity("fixture","b0:fixture:grid:0","native.fixture"))].mask)
        # Unit ownership remains present for normal A0 training.
        self.assertEqual(len(provider.associations),1)
        import torch
        from types import SimpleNamespace
        from data import DataAdapter
        from data.action4_dataset import prepare_rois
        rois,aff=prepare_rois([self.rgb],[obj["bbox_full_xyxy"]],[0],roi_size=448,expansion=2.25,device="cpu")
        down,_=DataAdapter.letterbox_affine((448,448),(112,112));A112=down@aff[0]
        unit={"metadata":{"crop_box":obj["bbox_full_xyxy"],"affine_hm":A112.tolist()},
            "augmentation_seed":1,"roi":rois[0],"geometry":{"K":torch.tensor(A112@np.asarray(K),dtype=torch.float32)},
            "active":True,"unit_marker":"original_A0_unit"}
        def identity(image,instance,roi,pixel):
            return SimpleNamespace(image_id=image,instance_id=instance,roi_id=roi,pixel_space_id=pixel)
        enriched=provider.enrich_training({"record":record,"instance":0,"image":self.rgb},unit,
            identity_factory=identity,mask_aux_factory=lambda **kw:SimpleNamespace(**kw))
        self.assertEqual(enriched["mask_aux"].available.sum().item(),0)
        self.assertIs(enriched["roi"],unit["roi"])
        self.assertEqual(enriched["unit_marker"],"original_A0_unit")
        self.assertTrue(enriched["sam_quality_supervised"])
        # Returned per-unit metadata must not mutate the provider's frozen ownership.
        frozen_owner=copy.deepcopy(provider.associations[("fixture","1")])
        enriched["sam_input"]["association"]["prediction_id"]="tampered_returned_owner"
        enriched["sam_input"]["association"]["qualified_prediction_ids"].clear()
        self.assertEqual(provider.associations[("fixture","1")],frozen_owner)
        again=provider.enrich_training({"record":record,"instance":0,"image":self.rgb},unit,
            identity_factory=identity,mask_aux_factory=lambda **kw:SimpleNamespace(**kw))
        self.assertEqual(again["sam_input"]["association"],frozen_owner)


    def test_reordering_and_gt_masks_do_not_choose_input(self):
        record,spec,index,binding,assoc=self.build()
        policy=assoc["policy"]
        modified=copy.deepcopy(record);modified["instances"][0].update(mask_visible="never_read_gt1.png",corners_px=[[999,999]]*8)
        annpath=self.base/"other_annotations.jsonl";annpath.write_text(json.dumps(modified)+"\n")
        other=build_associations(spec,{"path":str(annpath.relative_to(R)),"sha256":sha(annpath)},policy,self.base/"other_assoc.json")
        revised=json.loads((R/other["path"]).read_text())
        self.assertEqual(assoc["rows"],revised["rows"])
        self.assertEqual(assoc["cache_manifest"],revised["cache_manifest"])
        saved=json.loads((R/assoc["predictions"]["path"]).read_text())[0]["instances"]
        images={"fixture":index["images"][0]};statuses={("fixture",f"b0:fixture:grid:{i}"):"available" for i in range(3)}
        a,_=derive_rows([record],{"fixture":saved},images,statuses)
        b,_=derive_rows([record],{"fixture":list(reversed(saved))},images,statuses)
        self.assertEqual(a,b);self.assertEqual(a[0]["prediction_id"],"b0:fixture:grid:0")
    def test_prediction_score_not_gt_iou_or_native_availability_ranks(self):
        record,_=self.annotation()
        saved=[{"grid_index":0,"score":.8,"bbox_full_xyxy":[1.,1.,4.,5.]},
               {"grid_index":1,"score":.9,"bbox_full_xyxy":[1.3,1.,4.3,5.]}]
        images={"fixture":{"predictions":[{"prediction_id":f"b0:fixture:grid:{i}"} for i in range(2)]}}
        statuses={("fixture","b0:fixture:grid:0"):"available",("fixture","b0:fixture:grid:1"):"ambiguous"}
        rows,_=derive_rows([record],{"fixture":saved},images,statuses)
        self.assertEqual(rows[0]["prediction_id"],"b0:fixture:grid:1")
        self.assertEqual(rows[0]["native_status"],"ambiguous")
        # Finite GT mask/corner supervision changes do not participate in ownership.
        record["instances"][0]["mask_visible"]="another_loss_only_mask.png"
        record["instances"][0]["corners_px"]=[[0.,0.]]*8
        changed,_=derive_rows([record],{"fixture":list(reversed(saved))},images,statuses)
        self.assertEqual(rows,changed)

    def test_partial_cache_and_legacy_production_rejected(self):
        self.progress["completed_jobs"]=1;self.jsonfile(self.native/"PROGRESS.json",self.progress)
        with self.assertRaisesRegex(ValueError,"Incomplete"):self.do_export()
    def test_legacy_v1_explicit_synthetic_only(self):
        from data.action4_sam_roi_inputs import SOURCE_KEYS,cache_key
        _,spec,index=self.do_export()
        old={"schema":"action4.sam_training_associations.v1","cache_manifest_sha256":spec["manifest"]["sha256"],
             "assignment_policy":ASSIGNMENT,"rows":[{"image_id":"fixture","gt_instance_id":1,"prediction_id":"b0:fixture:grid:0","status":"matched","reason":"explicit engineering fixture"}]}
        oldbinding=self.jsonfile(self.base/"old_assoc.json",old)
        with self.assertRaisesRegex(ValueError,"production association v2"):
            SamCacheProvider(spec,split="train",training_associations=oldbinding,allow_synthetic=True)
        # Distinct synthetic index and metadata, with explicit permission.
        synthetic=copy.deepcopy(index);synthetic.pop("producer_receipt")
        synthetic["origin"]="synthetic_fixture";synthetic["sources"]={key:"a"*64 for key in SOURCE_KEYS}
        for entry in synthetic["images"]:
            for i,b in enumerate(entry["predictions"]):
                with np.load(R/b["path"],allow_pickle=False) as z:
                    meta=json.loads(str(z["metadata"]));arrays={} if "mask" not in z else {"mask":z["mask"].copy()}
                meta.update(origin="synthetic_fixture",sources=synthetic["sources"]);meta["cache_key"]=cache_key(meta)
                path=self.base/f"synthetic_{i}.npz";np.savez(path,metadata=np.array(json.dumps(meta)),**arrays)
                b.update(path=str(path.relative_to(R)),sha256=sha(path))
        manifest=self.jsonfile(self.base/"synthetic_index.json",synthetic)
        old["cache_manifest_sha256"]=manifest["sha256"];oldbinding=self.jsonfile(self.base/"synthetic_assoc.json",old)
        ss={"manifest":manifest,"expected_sources":synthetic["sources"]}
        self.assertEqual(len(SamCacheProvider(ss,split="train",training_associations=oldbinding,allow_synthetic=True).associations),1)
        with self.assertRaises(ValueError):SamCacheProvider(ss,split="train",training_associations=oldbinding)

if __name__=="__main__":unittest.main()
