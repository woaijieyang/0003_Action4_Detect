import unittest
from model.action4_sam_seed_output import merge_seed_output

class SeedOutputTests(unittest.TestCase):
    def test_fresh_frame_preserves_computed_mask(self):
        mask=object()
        self.assertIs(merge_seed_output({},4,{0:mask})[0],mask)
    def test_existing_frame_preserves_other_ids_without_mutation(self):
        a,b=object(),object();cache={4:{1:a}}
        result=merge_seed_output(cache,4,{0:b})
        self.assertEqual(set(result),{0,1});self.assertEqual(set(cache[4]),{1})
    def test_empty_refinement_does_not_invent_target(self):
        self.assertEqual(merge_seed_output({},4,None),{})
    def test_none_mask_rejected(self):
        with self.assertRaises(ValueError):merge_seed_output({},4,{0:None})
    def test_same_id_refinement_replaces_only_output(self):
        a,b=object(),object();cache={4:{0:a}}
        self.assertIs(merge_seed_output(cache,4,{0:b})[0],b)
        self.assertIs(cache[4][0],a)


class SeedBridgeTests(unittest.TestCase):
    def test_only_fresh_frame_uses_new_mask(self):
        from model.action4_sam_seed_output import make_seed_output_bridge
        calls=[]
        def old(state,frame,refined=None):
            calls.append(frame)
            return {"original":True}
        bridge=make_seed_output_bridge(old)
        mask=object();state={"cached_frame_outputs":{}}
        self.assertEqual(bridge(state,4,{0:mask}),{0:mask})
        self.assertEqual(calls,[])
        state["cached_frame_outputs"][4]={}
        self.assertEqual(bridge(state,4,{0:mask}),{"original":True})
        self.assertEqual(calls,[4])
    def test_no_computed_mask_delegates_without_fake_object(self):
        from model.action4_sam_seed_output import make_seed_output_bridge
        bridge=make_seed_output_bridge(lambda *args:{})
        self.assertEqual(bridge({"cached_frame_outputs":{}},4,None),{})
