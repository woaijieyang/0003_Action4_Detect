"""CPU arithmetic and checkpoint tests; no distributed process or training launch."""
import copy
import json
import unittest
from unittest.mock import Mock
import torch
from data.action4_sam_roi_routing import ROUTE_COUNT_FIELDS
from model.training.action4_route_counts import (empty_counts,count_vector,counts_from_vector,
    local_route_counts,reduce_route_update,advance_counts,checkpoint_route_counts,
    restore_route_counts,route_log_fields)
from model.training.action4_supervision import has_supervision


def item(requested,effective=None,fallback=None,active=True):
    return {"active":active,"metadata":{"sam_roi_routing":{
        "active":active,"requested_route":requested,
        "effective_route":requested if effective is None else effective,"fallback_reason":fallback}}}


class RouteCounterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_fixed_world2_uneven_micro_dummy_counts(self):
        items=[item("gt_jitter"),item("prediction_coarse","prediction_full","requested_route_unavailable"),
               item("prediction_full","gt_jitter","predicted_roi_missing"),item("extra_offset"),item("prediction_full")]
        full=local_route_counts(items)
        ranks=[items[:3],items[3:]+[item("gt_jitter",active=False)]]
        vectors=[count_vector(local_route_counts(rank)) for rank in ranks]
        summed=[a+b for a,b in zip(*vectors)]
        self.assertEqual(counts_from_vector(summed),full)
        self.assertEqual(len(summed),11);self.assertEqual(tuple(ROUTE_COUNT_FIELDS)[0],"active_units")
        for rank in ranks:
            micros=[rank[:2],rank[2:]]
            accumulated=[0]*11
            for micro in micros:
                accumulated=[a+b for a,b in zip(accumulated,count_vector(local_route_counts(micro)))]
            self.assertEqual(accumulated,count_vector(local_route_counts(rank)))
        for i,rank in enumerate(ranks):
            calls=[]
            def reducer(tensor):
                calls.append(1);tensor.add_(torch.tensor(vectors[1-i],dtype=torch.int64))
            actual=reduce_route_update(rank,enabled=True,device="cpu",all_reduce=reducer)
            self.assertEqual(actual,full);self.assertEqual(calls,[1])
        self.assertEqual(full["active_units"],5)
        self.assertEqual(full["fallbacks"],{"requested_route_unavailable":1,"predicted_roi_missing":1})

    def test_default_disabled_has_no_collective_or_item_access(self):
        reducer=Mock(side_effect=AssertionError("no collective"))
        self.assertIsNone(reduce_route_update(None,enabled=False,device="unused-device",all_reduce=reducer))
        reducer.assert_not_called()
        self.assertEqual(local_route_counts([{"active":False}]),empty_counts())
        with self.assertRaises(ValueError):local_route_counts([{"active":True}])
        bad=item("unknown")
        with self.assertRaises(ValueError):local_route_counts([bad])

    def test_no_supervision_consumed_units_persist_and_resume_exactly(self):
        first=local_route_counts([item("gt_jitter"),item("extra_offset")])
        self.assertFalse(has_supervision({"xyz":0},{"xyz":1}))
        original=empty_counts()
        cumulative=advance_counts(original,first,epoch=0,cursor=0,units_per_epoch=3)
        self.assertEqual(original,empty_counts())
        state=checkpoint_route_counts(cumulative,epoch=0,cursor=2,units_per_epoch=3)
        # JSON-compatible state is also safe in the existing torch checkpoint.
        restored=restore_route_counts(json.loads(json.dumps(state)),epoch=0,cursor=2,units_per_epoch=3)
        second=local_route_counts([item("prediction_coarse","prediction_full","requested_route_unavailable")])
        after=advance_counts(restored,second,epoch=0,cursor=2,units_per_epoch=3)
        final=checkpoint_route_counts(after,epoch=1,cursor=0,units_per_epoch=3)
        resumed=restore_route_counts(final,epoch=1,cursor=0,units_per_epoch=3)
        self.assertEqual(resumed["active_units"],3)
        self.assertEqual(resumed,local_route_counts([item("gt_jitter"),item("extra_offset"),
             item("prediction_coarse","prediction_full","requested_route_unavailable")]))
        with self.assertRaises(ValueError):restore_route_counts(state,epoch=0,cursor=0,units_per_epoch=3)
        with self.assertRaises(ValueError):advance_counts(restored,first,epoch=0,cursor=0,units_per_epoch=3)
        with self.assertRaises(ValueError):restore_route_counts(None,epoch=0,cursor=2,units_per_epoch=3)
        with self.assertRaises(ValueError):restore_route_counts(state,epoch=0,cursor=2,units_per_epoch=4)
        # Replayed log entry has same logical key and cumulative snapshot, not added twice.
        a=route_log_fields(first,cumulative,epoch=0,cursor_after=2,units_per_epoch=3)
        b=route_log_fields(first,restored,epoch=0,cursor_after=2,units_per_epoch=3)
        self.assertEqual(a,b)
        self.assertEqual(a["roi_route_progress"]["logical_update_key"],"0:0:2")
        a["roi_route_counts_cumulative"]["active_units"]=999
        self.assertEqual(cumulative["active_units"],2)
        next_update=local_route_counts([item("prediction_full")])
        next_counts=advance_counts(resumed,next_update,epoch=1,cursor=0,units_per_epoch=3)
        self.assertEqual(next_counts["active_units"],4)

    def test_counter_mass_corruption_refused(self):
        counts=local_route_counts([item("gt_jitter")])
        bad=copy.deepcopy(counts);bad["active_units"]=2
        with self.assertRaises(ValueError):count_vector(bad)
        state=checkpoint_route_counts(counts,epoch=0,cursor=1,units_per_epoch=3)
        bad=copy.deepcopy(state);bad["field_order"].reverse()
        with self.assertRaises(ValueError):restore_route_counts(bad,epoch=0,cursor=1,units_per_epoch=3)
        with self.assertRaises(ValueError):checkpoint_route_counts(counts,epoch=0,cursor=2,units_per_epoch=3)


if __name__=="__main__":unittest.main(verbosity=2)
