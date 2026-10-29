import copy,importlib,unittest
def row(frame,missing=(),untrusted=()):
 tracks=[]
 for i,tid in enumerate(('left','right')):
  box=[20.+i*90,20.,70.+i*90,60.]
  cid=f'b0:frame{frame}:grid:{i}'
  d={'candidate_id':cid,'bbox_full_xyxy':box,'score':.95 if tid not in untrusted else .5,'refined_quality_score':.95}
  c={'candidate_id':cid,'raw_B0_box':box,'score':d['score'],'refined_quality_score':.95,'accepted':tid not in untrusted}
  tracks.append({'track_id':tid,'status':'sam_supported' if tid in missing else 'observed','candidate_id':None if tid in missing else cid,'raw_detection':None if tid in missing else d,'geometry_trace':{'raw_calibration_evidence':c}})
 return {'frame':{'global_frame_id':frame},'fused':{'frame_id':frame,'tracks':tracks,'mode':'video_inference'},'native_unavailable':[]}
class ReanchorTests(unittest.TestCase):
 def gate(self):
  import importlib.util
  self.assertIsNotNone(importlib.util.find_spec('model.action4_sam_body_reanchor'),'body reanchor state machine missing')
  m=importlib.import_module('model.action4_sam_body_reanchor')
  return m.BodyReanchorGate({'left':0,'right':1},source_session_id='old')
 def prep(self,g):
  for f in range(5):self.assertEqual(g.observe(row(f))['action'],'observe')
 def test_causal_recent_seed_after_three_misses(self):
  g=self.gate();self.prep(g)
  for f in [5,6]:self.assertEqual(g.observe(row(f,('right',)))['action'],'observe')
  p=g.observe(row(7,('right',)))
  self.assertEqual(p['action'],'propose_reanchor');self.assertEqual(p['seed_frame'],4)
  self.assertEqual(p['trigger_frame'],7);self.assertEqual(p['responsibility_start'],7)
  self.assertEqual(p['confirmed_frames'],{'left':[0,1,2,3,4],'right':[0,1,2,3,4]})
  self.assertEqual(p['native_obj_map'],{'left':0,'right':1})
  self.assertEqual(g.observe(row(8,('right',)))['action'],'await_native_validation')
 def test_no_future_seed_and_no_old_anchor_after_gap(self):
  g=self.gate();self.prep(g)
  for f in range(5,23):g.observe(row(f,(),('left','right')))
  for f in [23,24,25]:out=g.observe(row(f,('right',)))
  self.assertEqual(out['action'],'observe')
 def test_ambiguous_or_inference_error_is_not_missing_target(self):
  for status in ['inference_error','ambiguous_identity','session_lost','missing_response']:
   g=self.gate();self.prep(g)
   for f in [5,6,7]:
    r=row(f,('right',));r['native_state']=status;out=g.observe(r)
   self.assertEqual(out['action'],'no_automatic_reanchor')
 def test_generation_commit_requires_exact_new_session_binding(self):
  g=self.gate();self.prep(g)
  for f in [5,6,7]:p=g.observe(row(f,('right',)))
  good={'session_id':'new','generation':1,'frame_id':7,'native_obj_map':{'left':0,'right':1},'status':'native_binding_validated'}
  for key,val in [('session_id','old'),('generation',0),('frame_id',6),('native_obj_map',{'left':1,'right':0})]:
   b=copy.deepcopy(good);b[key]=val
   with self.assertRaises(ValueError):g.resolve(p['event_id'],binding=b)
  g.resolve(p['event_id'],binding=good)
  self.assertEqual(g.generation,1);self.assertEqual(g.session_id,'new')
 def test_failed_correction_preserves_current_session_and_generation(self):
  g=self.gate();self.prep(g)
  for f in [5,6,7]:p=g.observe(row(f,('right',)))
  g.resolve(p['event_id'],binding=None,failure='native_oom')
  self.assertEqual(g.generation,0);self.assertEqual(g.session_id,'old')
 def test_duplicate_frames_and_swapped_candidate_evidence_rejected(self):
  g=self.gate();g.observe(row(0))
  with self.assertRaises(ValueError):g.observe(row(0))
  g=self.gate();r=row(0);r['fused']['tracks'][0]['geometry_trace']['raw_calibration_evidence']['candidate_id']='wrong'
  with self.assertRaises(ValueError):g.observe(r)
 def test_discontinuous_frames_do_not_keep_confirmations(self):
  g=self.gate();self.prep(g)
  for f in [10,11,12]:out=g.observe(row(f,('right',)))
  self.assertEqual(out['action'],'observe')
 def test_overlapping_distinct_boxes_are_not_merged(self):
  g=self.gate()
  for f in range(5):
   r=row(f)
   t=r['fused']['tracks'][1];t['raw_detection']['bbox_full_xyxy']=[25.,20.,75.,60.];t['geometry_trace']['raw_calibration_evidence']['raw_B0_box']=[25.,20.,75.,60.]
   g.observe(r)
  for f in [5,6,7]:p=g.observe(row(f,('right',)))
  self.assertEqual(set(p['seed_tracks']),{'left','right'})
if __name__=='__main__':unittest.main()
