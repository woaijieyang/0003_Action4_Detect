import copy,json,unittest
from pathlib import Path
import numpy as np
from eval.action4_native_tracking_audit import audit_records
from eval.action4_native_candidate_bridge import join_candidate_rows

ROOT=Path('/data/zy/vln/a-test')


def saved_case(relative):
    p=ROOT/relative
    if not (p/'TRACK_COMPLETE.json').is_file():raise RuntimeError('Actual completed native fixture required')
    m=json.loads((p/'INPUT_MANIFEST.json').read_text());n=[json.loads(x) for x in (p/'POSES.jsonl').read_text().splitlines()];e=json.loads((p/'FRAME_EVENTS.json').read_text())
    return m,n,e


def corrected():return saved_case('outputs/action4_final_pose_native_bridge_v002/corrected_payload')


def update_decision_record(n,e,frame_index,track_index):
    d=e[frame_index]['recovery_decisions'][track_index]
    next(r for r in n if r['frame_id']==d['frame_id'] and r['track_id']==d['track_id'])['recovery_decision']=copy.deepcopy(d)


class SelectedAuditTests(unittest.TestCase):
    def test_actual_selected_closed_records(self):
        m,n,e=corrected();a=audit_records(m,n,e)
        self.assertEqual(a['native_pose_input_kind'],'selected_A0');self.assertFalse(a['raw_pose_consumed_for_initialization']);self.assertTrue(a['raw_baseline_preserved']);self.assertEqual(a['records'],6)
    def test_actual_legacy_raw_closed_records(self):
        m,n,e=saved_case('outputs/action4_m3t_pose_cache_v001/parity3');a=audit_records(m,n,e)
        self.assertEqual(a['native_pose_input_kind'],'raw_A0');self.assertTrue(a['raw_pose_consumed_for_initialization'])
    def test_selected_fallback_raw_source_and_route_not_interchangeable(self):
        for kind in ['fallback','raw_pose','raw_corners','route','init_event']:
            m,n,e=corrected();target=m['rows'][0]['instances'][0]
            if kind=='fallback':n[0]['final_corners']=copy.deepcopy(target['raw_corners'])
            if kind=='raw_pose':n[0]['raw_baseline_pose']['translation_m'][0]+=.01
            if kind=='raw_corners':n[0]['raw_baseline_corners'][0][0]+=1
            if kind=='route':n[0]['native_pose_input_kind']='raw_A0'
            if kind=='init_event':next(c for c in e[0]['calls'] if c['name']=='initialize_selected_A0_once')['name']='initialize_raw_A0_once'
            with self.assertRaises(ValueError,msg=kind):audit_records(m,n,e)
    def test_unknown_recovery_mode_rejected(self):
        m,n,e=corrected();m['recovery_mode']='unknown'
        with self.assertRaises(ValueError):audit_records(m,n,e)
    def test_record_only_rejects_actual_or_unlogged_pose_reset(self):
        for kind in ['action','call','pose']:
            m,n,e=corrected();d=e[1]['recovery_decisions'][0]
            if kind=='action':d.update(action='reseed',native_restart_succeeded=True)
            if kind=='call':e[1]['calls'].append(dict(name='RestartSupportedBody_current_evidence',track_id=d['track_id'],ok=True))
            if kind=='pose':d['native_pose_after_decision'][0][3]+=.01
            update_decision_record(n,e,1,0)
            with self.assertRaises(ValueError,msg=kind):audit_records(m,n,e)
    def test_selected_reseed_must_match_selected_seed(self):
        m,n,e=corrected();mode='current_evidence_optional_reseed_and_SAM_only_continuation';m['recovery_mode']=mode
        for i,event in enumerate(e):
            for j,d in enumerate(event['recovery_decisions']):d['recovery_mode']=mode;update_decision_record(n,e,i,j)
        d=e[-1]['recovery_decisions'][0];target=m['rows'][-1]['instances'][0];p=target['selected_A0_pose'];T=np.eye(4);T[:3,:3]=p['rotation'];T[:3,3]=p['translation_m'];T=T.astype(np.float32).astype(float).tolist()
        d.update(action='reseed',native_restart_succeeded=True,reason='two_frame_contradiction_with_current_consistent_A0_B0_SAM',recovery_id=1,native_pose_after_decision=T)
        e[-1]['calls'].append(dict(name='RestartSupportedBody_current_evidence',track_id=d['track_id'],ok=True));update_decision_record(n,e,2,0)
        row=next(r for r in n if r['frame_id']==d['frame_id'] and r['track_id']==d['track_id']);row.update(raw_A0_reset_this_frame=True,native_pose=T)
        self.assertEqual(audit_records(m,n,e)['recovery_count'],1)
        p=target['raw_A0_pose'];T=np.eye(4);T[:3,:3]=p['rotation'];T[:3,3]=p['translation_m'];d['native_pose_after_decision']=T.astype(np.float32).astype(float).tolist();update_decision_record(n,e,2,0)
        with self.assertRaises(ValueError):audit_records(m,n,e)
    def test_actual_selected_join_preserves_all_base_fields(self):
        m,n,e=corrected();wanted=set(m['frame_ids'])
        with (ROOT/m['baseline_predictions']['path']).open() as f:base=[r for line in f if (r:=json.loads(line))['frame_id'] in wanted]
        before=copy.deepcopy(base);joined=join_candidate_rows(base,n,m)
        self.assertEqual(base,before)
        for row in joined:
            for inst in row['prediction']['instances']:
                candidate=inst.pop('native_tracking_candidate');self.assertEqual(candidate['native_pose_input_kind'],'selected_A0');self.assertFalse(candidate['deployment_admitted'])
        self.assertEqual(joined,before)
    def test_suppressed_reseed_not_numerically_promoted(self):
        m,n,e=corrected();wanted=set(m['frame_ids'])
        with (ROOT/m['baseline_predictions']['path']).open() as f:base=[r for line in f if (r:=json.loads(line))['frame_id'] in wanted]
        n[2]['recovery_decision']=dict(action='record_only_no_reseed')
        joined=join_candidate_rows(base,n,m)
        target=next(i for r in joined if r['frame_id']==n[2]['frame_id'] for i in r['prediction']['instances'] if i['track_id']==n[2]['track_id'])
        self.assertFalse(target['native_tracking_candidate']['numerical_candidate_eligible'])

class PairGeometryContractTests(unittest.TestCase):
    def test_actual_sealed_recipe_and_reject_identity_changes(self):
        from eval.action4_native_candidate_bridge import _geometry_recipe
        m,_,_=corrected();lock=json.loads((ROOT/m['input_lock']['path']).read_text());pair=json.loads((ROOT/lock['formal_pair_gate']['path']).read_text());asset=json.loads((ROOT/lock['CAD']['path']).read_text())
        roi,evidence=_geometry_recipe(pair,lock,asset);self.assertEqual(roi,1.5);self.assertEqual(evidence['kind'],'sealed_research_and_bound_trial_selection')
        wrong=copy.deepcopy(pair);wrong['models']['A0']['selected_step']+=1
        with self.assertRaises(ValueError):_geometry_recipe(wrong,lock,asset)
        bad=copy.deepcopy(asset);bad['corners_3d_mm'][0][0]+=1
        with self.assertRaises(ValueError):_geometry_recipe(pair,lock,bad)
    def test_sealed_readiness_flags_required(self):
        from unittest.mock import patch
        import eval.action4_native_candidate_bridge as bridge
        m,_,_=corrected();lock=json.loads((ROOT/m['input_lock']['path']).read_text());pair=json.loads((ROOT/lock['formal_pair_gate']['path']).read_text());asset=json.loads((ROOT/lock['CAD']['path']).read_text())
        read=bridge._read
        for key,value in [('status','PENDING'),('strict_load',False),('frozen_backbone_verified',False)]:
            def changed(ref):
                raw=read(ref)
                if ref==pair['strict_reload']:
                    obj=json.loads(raw);obj[key]=value;return json.dumps(obj)
                return raw
            with patch.object(bridge,'_read',side_effect=changed):
                with self.assertRaises(ValueError):bridge._geometry_recipe(pair,lock,asset)
    def test_legacy_geometry_checks_remain_and_unknown_missing_fields_reject(self):
        from eval.action4_native_candidate_bridge import _geometry_recipe
        lock={'CAD':{'path':'cad','sha256':'x'},'video':{'K':{'path':'K','sha256':'y'}}};pair={'roi_expansion':1.5,'CAD':lock['CAD'],'camera':lock['video']['K']}
        self.assertEqual(_geometry_recipe(pair,lock,{})[0],1.5)
        pair['camera']={'path':'other','sha256':'y'}
        with self.assertRaises(ValueError):_geometry_recipe(pair,lock,{})
        with self.assertRaises(ValueError):_geometry_recipe({'status':'unknown'},lock,{})
