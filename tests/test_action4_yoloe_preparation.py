import unittest
import numpy as np
from eval.action4_yolo_prompt_metrics import score_image,summarize
from model.action4_yoloe_adapter import rgb_to_bgr,PREDICT_ARGS
from data.action4_prompt_bank import mask_bbox,recipe_views

class PromptTests(unittest.TestCase):
    def setUp(self):
        self.a=np.zeros((40,60),bool);self.a[10:20,10:20]=True
        self.b=np.zeros((40,60),bool);self.b[10:20,22:32]=True
    def target(self,id,m):
        y,x=np.where(m);b=[x.min()-.5,y.min()-.5,x.max()+.5,y.max()+.5] if len(x) else [9.5,9.5,19.5,19.5]
        return dict(id=id,mask=m,bbox_full_xyxy=b,class_name='camera',annotation_status='valid')
    def pred(self,id,m,box=(7.5,7.5,21.5,21.5)):
        return dict(id=id,mask=m,bbox=([9.5,9.5,19.5,19.5] if m is None or not np.any(m) else [np.where(m)[1].min()-.5,np.where(m)[0].min()-.5,np.where(m)[1].max()+.5,np.where(m)[0].max()+.5]),prompt_bbox=box,source='yolo',class_id=0)
    def score(self,p,t):return score_image(p,t,image_id='fixture',class_protocol=dict(protocol_id='synthetic-v1',frozen=True,prediction_class_map={'0':'camera','1':'phone'},target_classes=['camera']))
    def test_single_wide(self):
        r=self.score([self.pred('p',self.a)],[self.target('a',self.a)])
        self.assertEqual(r['joint_tp'],1);self.assertTrue(r['strict_image_success'])
    def test_dual_overlapping_boxes_independent_masks(self):
        p=[self.pred('a',self.a,(5,5,24,24)),self.pred('b',self.b,(17,5,36,24))]
        r=self.score(p,[self.target('a',self.a),self.target('b',self.b)])
        self.assertEqual(r['joint_tp'],2);self.assertTrue(r['dual_success'])
    def test_merged_support(self):
        r=self.score([self.pred('m',self.a|self.b)],[self.target('a',self.a),self.target('b',self.b)])
        self.assertEqual(r['joint_tp'],0);self.assertEqual(r['joint_fn'],2)
    def test_duplicate_support(self):
        r=self.score([self.pred('a',self.a),self.pred('b',self.a)],[self.target('a',self.a)])
        self.assertEqual(r['joint_tp'],1);self.assertEqual(r['joint_fp'],1);self.assertFalse(r['strict_image_success'])
    def test_empty_predictions(self):
        r=self.score([],[self.target('a',self.a)])
        self.assertEqual(r['joint_fn'],1);self.assertFalse(r['strict_image_success'])
    def test_empty_gt_false_positive(self):
        r=self.score([self.pred('p',self.a)],[])
        self.assertEqual(r['joint_fp'],1);self.assertFalse(r['strict_image_success'])
    def test_empty_both_unknown_rates(self):
        r=self.score([],[]);s=summarize([r],required_groups=['other_camera'])
        self.assertTrue(r['strict_image_success']);self.assertIsNone(s['joint_recall'])
        self.assertFalse(s['joint_99_observed']);self.assertEqual(s['missing_groups'],['other_camera'])
    def test_missing_support(self):
        r=self.score([self.pred('p',None)],[self.target('a',self.a)])
        self.assertEqual(r['missing_support_count'],1);self.assertEqual(r['joint_tp'],0)
    def test_b0_rejected(self):
        p=self.pred('p',self.a);p['source']='b0'
        with self.assertRaises(ValueError):self.score([p],[self.target('a',self.a)])
    def test_permutation_stability(self):
        p=[self.pred('b',self.a),self.pred('a',self.a)]
        t=[self.target('t',self.a)]
        self.assertEqual(self.score(p,t)['joint_matches'],self.score(p[::-1],t)['joint_matches'])
    def test_wrong_prompt_independent_recognition(self):
        r=self.score([self.pred('p',self.a,(30,30,39,39))],[self.target('a',self.a)])
        self.assertEqual(r['recognition_tp'],1);self.assertEqual(r['joint_tp'],0)
    def test_rgb_once(self):
        x=np.array([[[1,2,3],[4,5,6]]],np.uint8)
        out=rgb_to_bgr(x);self.assertEqual(out.tolist(),[[[3,2,1],[6,5,4]]]);self.assertTrue(out.flags.c_contiguous)
        with self.assertRaises(ValueError):rgb_to_bgr(np.zeros((2,2,4),np.uint8))
        self.assertIsNone(PREDICT_ARGS['nms']);self.assertEqual(PREDICT_ARGS['imgsz'],1280)
    def test_reference_bbox(self):
        self.assertEqual(mask_bbox(self.a),[8.5,8.5,20.5,20.5])
        with self.assertRaises(ValueError):mask_bbox(np.zeros((10,10)))
    def test_view_recipe(self):
        views=recipe_views();self.assertEqual(len(views),48)
        self.assertEqual(len({v['reference_id'] for v in views}),48)
        self.assertEqual(sorted(v['active_name'] for v in views if v['active_name']),['back','backupper','front','frontupper','left','right'])

    def test_other_camera_support_pollution(self):
        t=[self.target('a',self.a),dict(self.target('other_camera',self.b),class_name='other_camera')]
        r=self.score([self.pred('p',self.a|self.b)],t)
        self.assertEqual(r['gt_count'],1);self.assertEqual(r['joint_tp'],0)
    def test_other_camera_not_target_denominator(self):
        t=[self.target('a',self.a),dict(self.target('other',self.b),class_name='other_camera')]
        r=self.score([self.pred('p',self.a)],t)
        self.assertEqual(r['gt_count'],1);self.assertEqual(r['joint_tp'],1)
    def test_exact_bound(self):
        from eval.action4_yolo_prompt_metrics import independent_unit_interval
        self.assertAlmostEqual(independent_unit_interval(100,100)['lower_bound_95'],.05**.01)
        self.assertLess(independent_unit_interval(298,298)['lower_bound_95'],.99)
        self.assertGreaterEqual(independent_unit_interval(299,299)['lower_bound_95'],.99)
        self.assertIsNone(independent_unit_interval(0,0)['rate'])
    def test_bad_group_blocks_global_gate(self):
        p=[self.pred('a',self.a,(5,5,24,24)),self.pred('b',self.b,(17,5,36,24))]
        t=[self.target('a',self.a),self.target('b',self.b)]
        records=[]
        for i in range(200):
            r=self.score(p,t);r['image_id']=str(i);records.append(r)
        r=self.score([],t);r['image_id']='bad';r['groups']=['hard'];records.append(r)
        result=summarize(records,required_groups=['hard'])
        self.assertGreater(result['joint_recall'],.99);self.assertFalse(result['joint_99_observed'])


    def test_full_occlusion_kept_recognition(self):
        t=self.target('a',np.zeros_like(self.a));p=self.pred('p',self.a)
        r=self.score([p],[t])
        self.assertEqual(r['gt_count'],1);self.assertEqual(r['recognition_tp'],1)
        self.assertEqual(r['support_gt_count'],0);self.assertIsNone(r['strict_image_success'])
        self.assertTrue(r['recognition_strict_image_success'])
    def test_unobserved_gt_not_empty_negative(self):
        t=self.target('a',np.zeros_like(self.a));r=self.score([],[t])
        self.assertEqual(r['recognition_fn'],1);self.assertFalse(r['recognition_strict_image_success'])
        self.assertIsNone(r['strict_image_success'])
    def test_missing_full_box_unknown(self):
        t=self.target('a',self.a);del t['bbox_full_xyxy'];r=self.score([],[t])
        self.assertEqual(r['recognition_unmeasured_gt_ids'],['a']);self.assertIsNone(r['strict_image_success'])
    def test_ignored_not_true_negative(self):
        t=self.target('a',self.a);t['annotation_status']='ignore';r=self.score([],[t])
        self.assertEqual(r['ignored_gt_ids'],['a']);self.assertIsNone(r['recognition_strict_image_success'])
    def test_box_recognition_without_mask(self):
        r=self.score([self.pred('p',None)],[self.target('a',self.a)])
        self.assertEqual(r['recognition_tp'],1);self.assertEqual(r['support_tp'],0)
    def test_class_correct_rejected(self):
        p=self.pred('p',self.a);p['class_correct']=True
        with self.assertRaises(ValueError):self.score([p],[self.target('a',self.a)])
    def test_missing_protocol_rejected(self):
        with self.assertRaises(ValueError):score_image([],[],image_id='x',class_protocol={})
    def test_wrong_class_no_target_credit(self):
        p=self.pred('p',self.a);p['class_id']=1;r=self.score([p],[self.target('a',self.a)])
        self.assertEqual(r['recognition_tp'],0);self.assertEqual(r['negative_prediction_ids'],['p'])
    def test_box_iou_failure_support_pass(self):
        p=self.pred('p',self.a);p['bbox']=[30,30,39,39]
        r=self.score([p],[self.target('a',self.a)])
        self.assertEqual(r['recognition_tp'],0);self.assertEqual(r['support_tp'],1)
    def test_recognition_dual_separate(self):
        pa=self.pred('a',self.a,(30,30,39,39));pb=self.pred('b',self.b,(30,30,39,39))
        r=self.score([pa,pb],[self.target('a',self.a),self.target('b',self.b)])
        self.assertTrue(r['recognition_dual_success']);self.assertTrue(r['independent_dual_success']);self.assertFalse(r['dual_success'])
    def test_registry_units_conservative(self):
        r=self.score([self.pred('p',self.a)],[self.target('a',self.a)])
        r['source_group']='sceneA';r['independent_unit_id']='unitA'
        registry=dict(frozen=True,protocol_id='unit-v1',units={'unitA':dict(source_group='sceneA',image_ids=['fixture'])})
        s=summarize([r],required_groups=[],unit_registry=registry)
        self.assertEqual(s['independent_units'],1);self.assertEqual(s['unit_strict_interval']['trials'],1);self.assertFalse(s['formal_99_pass'])
        r2=dict(r,image_id='frame2',independent_unit_id='unitB')
        registry['units']['unitB']=dict(source_group='sceneA',image_ids=['frame2'])
        with self.assertRaises(ValueError):summarize([r,r2],required_groups=[],unit_registry=registry)
    def test_prediction_validation(self):
        from model.action4_yoloe_adapter import validate_prediction_arrays
        b=np.array([[0,0,2,2]],float);sc=np.array([.5]);c=np.array([0]);m=np.ones((1,3,3))
        validate_prediction_arrays(b,sc,c,m,(3,3))
        for arrays in [(b,np.array([np.nan]),c,m),(b,sc,np.array([np.nan]),m),(b,sc,c,np.full((1,3,3),np.nan)),(b,np.array([]),c,m),(b,sc,c,None)]:
            with self.assertRaises(ValueError):validate_prediction_arrays(*arrays,(3,3))
    def test_evidence_semantics(self):
        from model.action4_yoloe_adapter import evidence_availability
        empty=evidence_availability([],text=True);negative=evidence_availability([2],text=True)
        self.assertEqual(empty['inference_response_available'],1);self.assertEqual(empty['target_evidence_available'],0)
        self.assertEqual(empty['negative_evidence_available'],0);self.assertEqual(negative['target_evidence_available'],0)
        self.assertEqual(negative['negative_evidence_available'],1)
    def test_dirty_pinned_source_rejected(self):
        from unittest.mock import patch
        from model.action4_yoloe_adapter import verify_source_tree
        with patch('model.action4_yoloe_adapter.contained_path',return_value=__import__('pathlib').Path('/data/zy/vln/a-test/upstream/ultralytics')):
            with patch('model.action4_yoloe_adapter.subprocess.check_output',side_effect=['abc\n',' M file.py\n']):
                with self.assertRaises(ValueError):verify_source_tree('upstream/ultralytics','abc')
    def test_bank_recipe_completeness(self):
        from data.action4_prompt_bank import validate_recipe,formal_bank_complete
        cfg=dict(gray_backgrounds=[64,128,192],active_views=['front','back','left','right','frontupper','backupper'])
        validate_recipe(cfg);self.assertFalse(formal_bank_complete(cfg,[],preview=False))
        with self.assertRaises(ValueError):validate_recipe(dict(cfg,gray_backgrounds=[128]))
        with self.assertRaises(ValueError):validate_recipe(dict(cfg,active_views=['front']))
        fake=[dict(v,preview=False,resolution=1024,samples=128,rgb=[]) for v in recipe_views()]
        self.assertFalse(formal_bank_complete(cfg,fake,preview=False))


    def test_unknown_prediction_not_false_positive(self):
        for status in ('unknown','ignore'):
            t=self.target('a',self.a);t['annotation_status']=status
            r=self.score([self.pred('p',self.a)],[t]);summary=summarize([r],required_groups=[])
            self.assertIsNone(r['recognition_fp']);self.assertIsNone(r['recognition_tp'])
            self.assertIsNone(summary['recognition_precision']);self.assertIsNone(summary['recognition_recall'])
            self.assertEqual(summary['published_prediction_count'],1);self.assertEqual(summary['scorable_prediction_count'],0)
            self.assertEqual(summary['excluded_prediction_count'],1);self.assertEqual(summary['unmeasured_images'],1)
            self.assertFalse(summary['joint_99_observed'])
    def test_unknown_mixed_image_denominators(self):
        good=self.score([self.pred('p',self.a)],[self.target('a',self.a)])
        t=self.target('a',self.a);t['annotation_status']='unknown'
        unknown=self.score([self.pred('p',self.a)],[t]);unknown['image_id']='unknown'
        result=summarize([good,unknown],required_groups=[])
        self.assertEqual(result['published_prediction_count'],2);self.assertEqual(result['scorable_prediction_count'],1)
        self.assertEqual(result['recognition_precision'],1);self.assertEqual(result['recognition_tp'],1)
        self.assertEqual(result['excluded_prediction_count'],1);self.assertFalse(result['joint_99_observed'])
    def test_duplicate_bad_box_cannot_steal_good_representative(self):
        bad=self.pred('a',self.a,(30,30,39,39));good=self.pred('b',self.a,(7.5,7.5,21.5,21.5))
        r=self.score([bad,good],[self.target('gt',self.a)])
        self.assertEqual(r['joint_tp'],1);self.assertEqual(r['joint_fp'],1)
        self.assertEqual(r['joint_matches'],[('gt','b')])
        bad['id']='z';good['id']='a'
        renamed=self.score([bad,good],[self.target('gt',self.a)])
        self.assertEqual(renamed['joint_tp'],r['joint_tp']);self.assertEqual(renamed['joint_fp'],r['joint_fp'])
    def test_duplicate_support_cannot_count_as_two_targets(self):
        p=[self.pred('a',self.a),self.pred('b',self.a,(19.5,7.5,33.5,21.5))]
        r=self.score(p,[self.target('a',self.a),self.target('b',self.b)])
        self.assertEqual(r['support_tp'],1);self.assertEqual(r['joint_tp'],1)
        self.assertFalse(r['independent_dual_success']);self.assertEqual(r['joint_fp'],1)

class RuntimeResourceContractTests(unittest.TestCase):
    def test_reject_cuda_before_model_load(self):
        from model.action4_yoloe_adapter import validate_runtime_resources
        for device in ('cuda:0','cuda','0',0,'mps'):
            with self.assertRaises(ValueError):validate_runtime_resources(dict(cpu_only=True,max_cpu_threads=4),device)
    def test_cpu_uses_locked_budget(self):
        from model.action4_yoloe_adapter import validate_runtime_resources
        self.assertEqual(validate_runtime_resources(dict(cpu_only=True,max_cpu_threads=3),'cpu'),3)
        with self.assertRaises(ValueError):validate_runtime_resources(dict(cpu_only=False,max_cpu_threads=4),'cuda:0')
    def test_reject_invalid_thread_budget(self):
        from model.action4_yoloe_adapter import validate_runtime_resources
        for value in (None,0,-1,5,True,2.5,'4'):
            with self.assertRaises(ValueError):validate_runtime_resources(dict(cpu_only=True,max_cpu_threads=value),'cpu')

class ReferenceSubsetTests(unittest.TestCase):
    def test_active_subset_keeps_formal_recipe(self):
        from data.action4_prompt_bank import recipe_views,select_reference_views
        before=recipe_views();ids=[v['reference_id'] for v in before if v['active_name']]
        selected=select_reference_views(ids)
        self.assertEqual(len(selected),6);self.assertEqual(recipe_views(),before)
        self.assertEqual([v['reference_id'] for v in selected],ids)
    def test_reject_duplicate_unknown_empty(self):
        from data.action4_prompt_bank import select_reference_views,recipe_views
        rid=recipe_views()[0]['reference_id']
        for ids in ([rid,rid],['unknown'],[],rid):
            with self.assertRaises(ValueError):select_reference_views(ids)
    def test_default_limit_preserved(self):
        from data.action4_prompt_bank import select_reference_views,recipe_views
        self.assertEqual(select_reference_views(),recipe_views()[:1])
        self.assertEqual(len(select_reference_views(limit=48)),48)

class GPU1ResourceContractTests(unittest.TestCase):
    def lock(self):
        return dict(schema='action4.yoloe_runtime_gpu1.v1',cpu_only=False,max_cpu_threads=4,max_gpu_memory_gib=8,torch_allocator_budget_gib=7,gpu_mapping=dict(physical_index=1,physical_uuid='GPU-a77a9d4c-7642-7112-698c-b02673d254cb',logical_device='cuda:0'))
    def test_approved_uuid_mapping(self):
        from unittest.mock import patch
        from model.action4_yoloe_adapter import validate_runtime_resources
        with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':self.lock()['gpu_mapping']['physical_uuid']}):
            self.assertEqual(validate_runtime_resources(self.lock(),'cuda:0'),4)
    def test_reject_ordinal_or_missing_environment(self):
        from unittest.mock import patch
        from model.action4_yoloe_adapter import validate_runtime_resources
        for value in ('','0','1','GPU-not-approved'):
            with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':value}):
                with self.assertRaises(ValueError):validate_runtime_resources(self.lock(),'cuda:0')
    def test_gpu0_and_wrong_logical_rejected(self):
        from unittest.mock import patch
        from model.action4_yoloe_adapter import validate_runtime_resources
        lock=self.lock()
        with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':lock['gpu_mapping']['physical_uuid']}):
            with self.assertRaises(ValueError):validate_runtime_resources(lock,'cuda:1')
            lock['gpu_mapping']['physical_index']=0
            with self.assertRaises(ValueError):validate_runtime_resources(lock,'cuda:0')

class OfficialMaskShapeBridgeTests(unittest.TestCase):
    def test_shape_only_bridge_and_counterexamples(self):
        from model.action4_yoloe_adapter import squeeze_official_mask_channel
        original=np.array([[[[0.],[.25]],[[1.],[.75]]]],np.float32)
        fixed=squeeze_official_mask_channel(original)
        self.assertEqual(fixed.shape,(1,2,2));self.assertTrue(np.shares_memory(original,fixed))
        self.assertTrue(np.array_equal(original.reshape(fixed.shape),fixed))
        for wrong in (np.zeros((1,2,2),np.float32),np.zeros((1,2,2,3),np.float32)):
            with self.assertRaises(ValueError):squeeze_official_mask_channel(wrong)

class VisibleSupportGateV4Tests(unittest.TestCase):
    def setUp(self):
        self.a=np.zeros((40,60),bool);self.a[10:20,10:20]=True
        self.b=np.zeros((40,60),bool);self.b[10:20,24:34]=True
        self.protocol=dict(protocol_id='fixture-v4',frozen=True,prediction_class_map={'0':'camera'},target_classes=['camera'])
    def target(self,id,mask,box):return dict(id=id,mask=mask,bbox_full_xyxy=box,class_name='camera',annotation_status='valid')
    def prediction(self,id,mask,box,prompt):return dict(id=id,source='yolo',class_id=0,mask=mask,bbox=box,prompt_bbox=prompt)
    def score(self,p,t,id='fixture'):return score_image(p,t,image_id=id,class_protocol=self.protocol)
    def pair(self):
        targets=[self.target('a',self.a,[9.5,9.5,19.5,19.5]),self.target('b',self.b,[23.5,9.5,33.5,19.5])]
        preds=[self.prediction('a',self.a,[0,0,60,40],[7.5,7.5,21.5,21.5]),self.prediction('b',self.b,[0,0,60,40],[21.5,7.5,35.5,21.5])]
        return preds,targets
    def test_wide_native_boxes_do_not_veto_valid_visible_support_gate(self):
        p,t=self.pair();r=self.score(p,t);s=summarize([r],required_groups=[])
        self.assertEqual(r['recognition_tp'],0);self.assertEqual(r['joint_tp'],2)
        self.assertEqual(s['recognition_recall'],0);self.assertEqual(s['joint_recall'],1.)
        self.assertEqual(s['joint_dual_rate'],1.);self.assertTrue(s['joint_99_observed'])
        self.assertFalse(s['formal_99_pass']);self.assertEqual(s['schema'],'action4.yolo_prompt_metrics.v4')
        self.assertEqual(r['unmatched_joint_prediction_ids'],[]);self.assertEqual(r['unmatched_recognition_prediction_ids'],['a','b'])
    def test_target_free_false_positive_is_not_invented_non_dji_class(self):
        p,_=self.pair();r=self.score(p[:1],[])
        self.assertEqual(r['joint_fp'],1);self.assertFalse(r['strict_image_success'])
        self.assertEqual(r['unmatched_joint_prediction_ids'],['a'])
        self.assertEqual(r['prediction_diagnostics'][0]['diagnostic_category'],'background_or_unlabeled')
        self.assertFalse(r['prediction_diagnostics'][0]['non_dji_class_asserted'])
    def test_duplicate_and_merged_support_evidence(self):
        p,t=self.pair();duplicate=dict(p[0],id='copy');r=self.score([p[0],duplicate],t[:1])
        self.assertEqual(r['joint_tp'],1);self.assertEqual(r['joint_fp'],1)
        self.assertEqual(r['duplicate_support_groups'],[['a','copy']]);self.assertEqual(r['unmatched_joint_prediction_ids'],['copy'])
        self.assertEqual(r['prediction_diagnostics'][1]['diagnostic_category'],'duplicate_target_support')
        merged=self.prediction('merged',self.a|self.b,[7,7,36,22],[7,7,36,22]);r=self.score([merged],t)
        self.assertEqual(r['joint_tp'],0);self.assertEqual(r['prediction_diagnostics'][0]['diagnostic_category'],'merged_support')
        self.assertEqual(len(r['prediction_diagnostics'][0]['merged_support_evidence']),2)
    def test_full_box_offset_with_correct_support_is_localization_diagnostic(self):
        p,t=self.pair();p[0]['bbox']=[40,25,50,35];r=self.score(p[:1],t[:1])
        self.assertEqual(r['joint_tp'],1);self.assertEqual(r['recognition_tp'],0)
        d=r['prediction_diagnostics'][0];self.assertEqual(d['diagnostic_category'],'full_box_localization_mismatch')
        self.assertTrue(d['joint_matched']);self.assertFalse(d['non_dji_class_asserted'])
    def test_joint_dual_gate_cannot_be_hidden_by_many_single_images(self):
        p,t=self.pair();records=[self.score(p[:1],t[:1],str(i)) for i in range(100)]
        bad=[dict(x) for x in p];bad[1]['prompt_bbox']=[40,25,50,35]
        r=self.score(bad,t,'dualbad');self.assertTrue(r['independent_dual_success']);self.assertFalse(r['dual_success']);records.append(r)
        s=summarize(records,required_groups=[]);self.assertGreaterEqual(s['joint_recall'],.99);self.assertGreaterEqual(s['strict_image_rate'],.99)
        self.assertEqual(s['independent_dual_rate'],1.);self.assertEqual(s['joint_dual_rate'],0.);self.assertFalse(s['joint_99_observed'])
    def test_missing_groups_and_unknown_evidence_still_block(self):
        p,t=self.pair();r=self.score(p,t);self.assertFalse(summarize([r],required_groups=['missing'])['joint_99_observed'])
        t[0]['annotation_status']='unknown';r=self.score(p,t)
        self.assertFalse(summarize([r],required_groups=[])['joint_99_observed']);self.assertIsNone(r['unmatched_joint_prediction_ids'])
        self.assertTrue(all(d['diagnostic_category']=='unmeasured_annotation_domain' for d in r['prediction_diagnostics']))
    def test_legacy_v3_records_require_explicit_rescoring(self):
        p,t=self.pair();r=self.score(p,t);r['score_schema']='action4.yolo_prompt_metrics.v3'
        with self.assertRaisesRegex(ValueError,'Legacy'):summarize([r],required_groups=[])

class DeclarativeTextContractTests(unittest.TestCase):
    def spec(self):
        texts=[('cloth','negative','clothing'),('action camera','positive','camera'),('object','neutral','object'),('hand','negative','hand'),('compact action camera','positive','camera')]
        return dict(schema='action4.yoloe.text_spec.v1',profile_id='synthetic-role-order',claim_scope='camera_proxy_specificity_unverified',classes=[dict(class_id=i,text=t,evidence_role=r,semantic_class=c) for i,(t,r,c) in enumerate(texts)])
    def test_noncontiguous_roles_are_not_hardcoded(self):
        from model.action4_yoloe_adapter import evidence_availability
        spec=self.spec();negative=evidence_availability([0],text=True,spec=spec);positive=evidence_availability([4],text=True,spec=spec)
        self.assertEqual(negative['target_evidence_available'],0);self.assertEqual(negative['negative_evidence_available'],1)
        self.assertEqual(positive['target_class_ids'],[1,4]);self.assertEqual(positive['negative_class_ids'],[0,3])
        self.assertEqual(positive['target_evidence_available'],1);self.assertEqual(positive['negative_evidence_available'],0)
        neutral=evidence_availability([2],text=True,spec=spec);self.assertEqual(neutral['target_evidence_available'],0);self.assertEqual(neutral['negative_evidence_available'],0)
    def test_publication_and_explicit_project_mapping_preserve_native_ids(self):
        from model.action4_yoloe_adapter import class_evidence_map,select_role_predictions,remap_prediction_semantics
        r=dict(classes=np.arange(5),scores=np.ones(5)*.8,boxes=np.tile([0.,0.,2.,2.],(5,1)),masks=np.ones((5,2,2)),class_evidence_map=class_evidence_map(text=True,spec=self.spec()),reference_id='fixture',profile_sha256='a'*64)
        positive=select_role_predictions(r,role='positive');negative=select_role_predictions(r,role='negative')
        self.assertEqual([p['class_id'] for p in positive],[1,4]);self.assertEqual([p['class_id'] for p in negative],[0,3])
        project=remap_prediction_semantics(positive,{'camera':0});self.assertEqual([p['class_id'] for p in project],[0,0]);self.assertEqual([p['native_class_id'] for p in project],[1,4])
    def test_invalid_class_declarations_rejected(self):
        from model.action4_yoloe_adapter import validate_text_spec
        import copy
        for change in ('id','role','duplicate','claim'):
            spec=copy.deepcopy(self.spec())
            if change=='id':spec['classes'][1]['class_id']=0
            elif change=='role':spec['classes'][1]['evidence_role']='assume_target'
            elif change=='duplicate':spec['classes'][1]['text']=spec['classes'][0]['text']
            else:spec['claim_scope']='verified_DJI_specific'
            with self.assertRaises(ValueError):validate_text_spec(spec)
    def test_profile_contract_freezes_roles_names_order_and_bytes(self):
        import json
        from pathlib import Path
        from model.action4_yoloe_adapter import text_spec_sha256,sha256,load_text_profile_contract
        folder=Path('/data/zy/vln/a-test/outputs/action4_stage2_preparation_v001/text_interface_v001/test_fixture');folder.mkdir(parents=True,exist_ok=True);profile=folder/'synthetic.npz';spec=self.spec()
        np.savez_compressed(profile,embeddings=np.zeros((1,5,512),np.float32),names=np.asarray([r['text'] for r in spec['classes']]),model=np.array('synthetic-fixture'))
        side=profile.with_suffix('.npz.contract.json');contract=dict(schema='action4.yoloe.text_profile_contract.v1',spec=spec,spec_sha256=text_spec_sha256(spec),profile_sha256=sha256(profile),assets={'detector':'a'*64,'text_encoder':'b'*64});side.write_text(json.dumps(contract));frozen_sha=sha256(side)
        binding=load_text_profile_contract(profile,expected_contract_sha256=frozen_sha,expected_assets=contract['assets']);self.assertEqual(binding['spec']['classes'][0]['evidence_role'],'negative')
        with self.assertRaises(ValueError):load_text_profile_contract(profile)
        contract['spec']['classes'][0]['evidence_role']='positive';contract['spec_sha256']=text_spec_sha256(contract['spec']);side.write_text(json.dumps(contract))
        with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):load_text_profile_contract(profile,expected_contract_sha256=frozen_sha)

if __name__=='__main__':unittest.main()
