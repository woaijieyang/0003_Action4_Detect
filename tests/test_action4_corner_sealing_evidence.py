"""Logic fixtures for the immutable-worker seal contract, not real calibration/model proof."""
import unittest,tempfile,copy
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
from eval import action4_corner_sealing as m
class CornerSealEvidenceTests(unittest.TestCase):
 def setUp(self):
  root=m.R/'outputs/action4_corner_sealing_preparation_v001';root.mkdir(exist_ok=True)
  self.tmp=tempfile.TemporaryDirectory(dir=root);self.out=Path(self.tmp.name);(self.out/'REFERENCE_REPORT.json').write_text('{}');(self.out/'snapshot.pt').write_bytes(b'NOT A MODEL: fixture')
  self.identity={'fixture_signature':True};ids=[str(i) for i in range(650)]
  self.cfg=dict(output=str(self.out.relative_to(m.R)),validation_receipt='validation.json',validation_receipt_sha256='1'*64,
   corner_eval_source={'path':'source.json','sha256':'2'*64},evaluation_counterpart_sha256='3'*64,
   sam_eval_cache={'manifest':{'path':'cache','sha256':'4'*64},'expected_sources':{'sam_model_sha256':'5'*64}},
   roi_expansion=1.5,roi_source='frozen_b0_prediction',b0_full_box_quality_receipt={'path':'q','sha256':'6'*64},corner_refined_quality_bridge={'path':'bridge','sha256':'7'*64})
  self.identity['parent_stage_fallback']={'path':'parent','sha256':'8'*64,'selected_step':1}
  decoder=m.canonical({'policy':m.POLICY,'source_sha256':'9'*64,'scope':'verified_code_binding'})
  self.cal=dict(schema='action4.corner_checkpoint_calibration.v1',quality_ready=True,quality_gate_passed_for_selection=True,
   run_binding=dict(checkpoint_sha256='a'*64,source_sha256='2'*64,predictions_sha256='b'*64,b0_model_sha256='3'*64,
    sam_model_sha256='5'*64,sam_cache_sha256='4'*64,data_sha256='c'*64,decoder_sha256=decoder),
   raw_report={'image_ids':ids},candidate_report={'image_ids':ids},policy=m.CALIBRATION_POLICY)
  report=dict(calibration_receipt={'path':'cal.json','sha256':'d'*64},prediction_artifact={'path':'pred','sha256':'b'*64},corner_quality_ready=True,corner_eval_engineering_fixture=False,evaluation_images=650)
  self.row=dict(step=25,report=report,evaluated_snapshot={'path':str((self.out/'snapshot.pt').relative_to(m.R)),'sha256':'a'*64},pose_protocol_eligible=True,deployment_eligible=True,report_artifact={'path':'report','sha256':'e'*64})
 def tearDown(self):self.tmp.cleanup()
 def checked(self,ref):
  p=ref['path']
  if p.endswith('REFERENCE_REPORT.json'):return {'corner_reference_identity':self.identity}
  if p=='validation.json':return {'splits':{'eval':{'manifest':'ann','sha256':'c'*64}}}
  if p=='source.json':return {'files_sha256':{'src/model/action4_single_frame_pose.py':'9'*64}}
  if p=='cal.json':return self.cal
  raise AssertionError(p)
 def invoke(self):
  with ExitStack() as stack:
   for name,value in [('_corner_identity',self.identity),('corner_result_signature',self.identity),('eligible',True),('checked_binding',self.out/'snapshot.pt'),('prepare_corner_selection',({'deployment_eligible':True},{}))]:
    stack.enter_context(patch.object(m,name,return_value=value))
   stack.enter_context(patch.object(m,'_v2_checked',side_effect=self.checked))
   stack.enter_context(patch.object(m,'_v2_read',return_value=[{'sample_id':str(i)} for i in range(650)]))
   return m.validate_corner_sealing_reports(self.cfg,[self.row],25)
 def test_complete_metadata_binds_decoder_and_parent(self):
  r=self.invoke();self.assertEqual(r['decoder_policy'],m.POLICY);self.assertEqual(r['global_branch'],'candidate');self.assertEqual(r['parent_stage_fallback'],self.identity['parent_stage_fallback'])
 def test_unready_cannot_be_sealed(self):
  self.row['deployment_eligible']=False
  with self.assertRaisesRegex(ValueError,'unready'):self.invoke()
 def test_changed_decoder_or_data_binding_rejected(self):
  for key in ['decoder_sha256','data_sha256','sam_cache_sha256']:
   old=self.cal['run_binding'][key];self.cal['run_binding'][key]='f'*64
   with self.assertRaises(ValueError):self.invoke()
   self.cal['run_binding'][key]=old
 def test_missing_difficult_image_domain_rejected(self):
  self.cal['raw_report']['image_ids']=self.cal['raw_report']['image_ids'][:-1]
  self.cal['candidate_report']['image_ids']=self.cal['raw_report']['image_ids']
  with self.assertRaisesRegex(ValueError,'domain'):self.invoke()
