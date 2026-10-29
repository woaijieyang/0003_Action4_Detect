
"""Small fabricated native-format fixtures test contracts, never claim SAM inference."""
import copy,json,unittest,uuid,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R
from data.action4_sam_native_export import export,sha,canonical,verify_export_receipt
from data.action4_sam_roi_inputs import native_key,NativeMaskIdentity
from data.action4_sam_cache_provider import SamCacheProvider

class CompletedCacheTests(unittest.TestCase):
    def setUp(self):
        self.split=getattr(self,"split","train")
        self.base=R/'outputs/action4_sam_completed_cache_preparation_v001'/uuid.uuid4().hex
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
    def test_full_candidates_selected_mask_and_train_association_boundary(self):
        out,spec,index=self.do_export();provider=SamCacheProvider(spec,split='train')
        frame,preds=provider.frame_predictions('fixture',self.rgb)
        self.assertEqual(len(preds),3)
        statuses=[preds[native_key(NativeMaskIdentity('fixture',f'b0:fixture:grid:{i}','native.fixture'))].metadata['status'] for i in range(3)]
        self.assertEqual(statuses,['available','ambiguous','missing'])
        first=preds[native_key(NativeMaskIdentity('fixture','b0:fixture:grid:0','native.fixture'))]
        np.testing.assert_array_equal(first.mask,self.mask)
        self.assertEqual(first.metadata['sam_binding']['session_id'],'session0')
        self.assertIsNone(provider.associations)
        with self.assertRaisesRegex(ValueError,'association'):provider.enrich_training({}, {},identity_factory=None,mask_aux_factory=None)
        self.assertEqual(verify_export_receipt(index)['training_association_status'],'not_provided')
    def test_wrong_rgb_split_and_unknown_candidate(self):
        _,spec,index=self.do_export();provider=SamCacheProvider(spec,split='train')
        bad=self.rgb.copy();bad[0,0,0]^=1
        with self.assertRaises(ValueError):provider.frame_predictions('fixture',bad)
        with self.assertRaises(ValueError):SamCacheProvider(spec,split='eval')
        with self.assertRaises(ValueError):provider.frame_predictions('different',self.rgb)
    def test_incomplete_and_limit_refused_before_output(self):
        self.progress['completed_jobs']=1;self.jsonfile(self.native/'PROGRESS.json',self.progress)
        with self.assertRaisesRegex(ValueError,'Incomplete'):export(self.native,self.base/'bad')
        self.assertFalse((self.base/'bad').exists())
        with self.assertRaisesRegex(ValueError,'entire domain'):export(self.native,self.base/'limited',1)
    def test_changed_source_refused(self):
        self.bundle.chmod(0o644);self.bundle.write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError,'bundle changed'):export(self.native,self.base/'bad')
    def test_resealed_missing_candidate_split_and_variant_rejected(self):
        out,spec,index=self.do_export();original=copy.deepcopy(index)
        for mode in ('missing','split','variant'):
            bad=copy.deepcopy(original)
            if mode=='missing':bad['images'][0]['predictions'].pop()
            elif mode=='split':bad['split']='eval'
            else:bad['rgb_variant']='brightness'
            receipt=json.loads((out/'PRODUCER_RECEIPT.json').read_text())
            receipt['index_payload_sha256']=canonical({k:v for k,v in bad.items() if k!='producer_receipt'})
            binding=self.jsonfile(self.base/(mode+'_receipt.json'),receipt);bad['producer_receipt']=binding
            with self.assertRaises(ValueError):verify_export_receipt(bad)
    def test_eval_full_domain_supported(self):
        self.split='eval';self.setUp()
        _,spec,index=self.do_export()
        provider=SamCacheProvider(spec,split='eval')
        self.assertEqual(len(provider.frame_predictions('fixture',self.rgb)[1]),3)
    def test_source_hash_memoized_but_changes_detected(self):
        from unittest.mock import patch
        _,spec,_=self.do_export()
        original=hashlib.file_digest;calls=[]
        def tracked(stream,*args,**kw):
            calls.append(str(stream.name));return original(stream,*args,**kw)
        with patch('hashlib.file_digest',side_effect=tracked):
            SamCacheProvider(spec,split='train')
        self.assertNotIn(str(self.bundle),calls)
        self.bundle.chmod(0o644);self.bundle.write_bytes(b'changed after export')
        with self.assertRaisesRegex(ValueError,'bundle changed'):SamCacheProvider(spec,split='train')
    def test_resealed_replaced_pixels_and_unprompted_available_rejected(self):
        from data.action4_sam_roi_inputs import cache_key,array_digest
        from data.action4_sam_native_export import _pixel_hash
        out,spec,index=self.do_export()
        original=copy.deepcopy(index);original_receipt=json.loads((out/'PRODUCER_RECEIPT.json').read_text())
        for number,kind in ((0,'replaced_pixels'),(2,'unprompted_available')):
            bad=copy.deepcopy(original);receipt=copy.deepcopy(original_receipt)
            binding=bad['images'][0]['predictions'][number]
            with np.load(R/binding['path'],allow_pickle=False) as data:
                meta=json.loads(str(data['metadata']))
                mask=data['mask'].copy() if 'mask' in data.files else self.mask.copy()
            if kind=='replaced_pixels':
                mask[0,0]=~mask[0,0]
            else:
                meta.update(status='available',reason=None,assigned_instance_id=meta['native_identity'][1],
                    candidate_overlap=1.,sam_binding={'session_id':'invented','obj_id':7,'generation':0,
                        'chosen_mask_id':'invented','match_basis':'invented'})
            meta['mask_sha256']=array_digest(mask);meta['cache_key']=cache_key(meta)
            replaced=self.base/(kind+'.npz');np.savez(replaced,metadata=np.array(json.dumps(meta)),mask=mask)
            binding.update(path=str(replaced.relative_to(R)),sha256=sha(replaced))
            mapping=next(x for x in receipt['mapping'] if x['prediction_id']==binding['prediction_id'])
            mapping['cache_sha256']=binding['sha256']
            receipt['index_payload_sha256']=canonical({k:v for k,v in bad.items() if k!='producer_receipt'})
            bad['producer_receipt']=self.jsonfile(self.base/(kind+'_receipt.json'),receipt)
            forged_spec={'manifest':self.jsonfile(self.base/(kind+'_index.json'),bad),'expected_sources':spec['expected_sources']}
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError,'payload|pixels|Unavailable'):
                    SamCacheProvider(forged_spec,split='train')
    def test_json_hash_read_replacement_rejected_on_miss_and_memo_hit(self):
        from unittest.mock import patch
        import data.action4_sam_native_export as exporter
        for memo in (False,True):
            original=self.base/('race_json_'+str(memo)+'.json');replacement=self.base/('replacement_'+str(memo)+'.json')
            original.write_text('{"value":1}');replacement.write_text('{"value":2}')
            binding={'path':str(original.relative_to(R)),'sha256':sha(original)}
            if memo:exporter._v2_checked(binding)
            real=exporter._v2_sha
            def swap(path):
                value=real(path)
                if Path(path).resolve()==original.resolve():replacement.replace(original)
                return value
            with patch.object(exporter,'_v2_sha',side_effect=swap):
                with self.assertRaisesRegex(ValueError,'replaced'):exporter._v2_checked(binding)
    def test_rgb_hash_decode_replacement_rejected_on_memo_hit(self):
        from unittest.mock import patch
        import data.action4_sam_native_export as exporter
        original=R/self.row['image_path'];replacement=self.base/'replacement.png'
        Image.fromarray(np.zeros_like(self.rgb)).save(replacement)
        exporter._v2_rgb(original,self.row['image_sha256'])
        real=exporter._v2_sha
        def swap(path):
            value=real(path)
            if Path(path).resolve()==original.resolve():replacement.replace(original)
            return value
        with patch.object(exporter,'_v2_sha',side_effect=swap):
            with self.assertRaisesRegex(ValueError,'replaced'):export(self.native,self.base/'race_export')
        self.assertFalse((self.base/'race_export/INDEX.json').exists())
    def test_official_file_change_after_export_rejected(self):
        _,spec,_=self.do_export()
        official=self.base/'source/upstream/sam3/sam3/test.py'
        official.chmod(0o644);official.write_text('changed official source');official.chmod(0o444)
        with self.assertRaisesRegex(ValueError,'Official sealed source'):
            SamCacheProvider(spec,split='train')

    def test_registration_mismatch_refused(self):
        marker=next((self.native/'jobs').glob('*/COMPLETED.json'));done=json.loads(marker.read_text())
        resultpath=R/done['result_path'];result=json.loads(resultpath.read_text())
        if not result['available']:
            marker=list((self.native/'jobs').glob('*/COMPLETED.json'))[1];done=json.loads(marker.read_text())
            resultpath=R/done['result_path'];result=json.loads(resultpath.read_text())
        result['generation']=2;self.jsonfile(resultpath,result)
        done['files_sha256'][done['result_path']]=sha(resultpath);self.jsonfile(marker,done)
        with self.assertRaisesRegex(ValueError,'registration'):export(self.native,self.base/'bad')

if __name__=='__main__':unittest.main()
