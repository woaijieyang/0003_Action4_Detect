import copy,json,os,unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from model.action4_b0_fusion import GuidedB0
from model.action4_b0_fusion_proposals import FusionFrameInput,build_current_proposals
from model.action4_guidance_contract import FrameIdentity,SourceIdentity,CandidateRecord
from model.action4_local_evidence import PredictedSupport
from model.action4_proposal_pool import candidate_union
from model.training.action4_stage2_core import construct
from data.action4_component_data import ComponentData
from data.action4_dataset import prepare_full_images

ROOT=Path('/data/zy/vln/a-test');OUT=ROOT/'outputs/action4_b0_current_proposals_v001'
class CurrentProposalsTests(unittest.TestCase):
    def test_current_full_image_pool_and_external_coverage_are_actual_forward_inputs(self):
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'),'');torch.set_num_threads(2);torch.manual_seed(607)
        registry=json.loads((ROOT/'configs/model/action4_best_registry_v003.json').read_text())['entries']['B0']
        state=torch.load(ROOT/registry['path'],map_location='cpu',weights_only=True)
        cfg=copy.deepcopy(state['config']);cfg.update(role='B0',mode='integration',parent_bundle=str(ROOT/registry['path']),parent_best_step=registry['selected_step'],b0_variant='b0_fusion_image',guidance_lr=1e-4,evaluation_scoring_policy='center')
        model,_=construct(cfg,'cpu');model.eval()
        data=ComponentData('outputs/action4_phase2_stage1_v001/data/PHRASE2_VALIDATED.json','eval','B0');item=data[0];record=item['record']
        x,A=prepare_full_images([item['image']],size=tuple(cfg['preprocess']['full_size']),device='cpu')
        import hashlib
        frame=FrameIdentity(record['sample_id'],0,'native_pilot_test',record['width'],record['height'],hashlib.sha256((ROOT/record['image']).read_bytes()).hexdigest())
        source=model.prediction_source
        inp=FusionFrameInput(frame,source,A[0]);calls=[];hook=model.detector.backbone.register_forward_hook(lambda *args:calls.append(1))
        try:dense,features,outputs,records,traces=model(x,frame_inputs=(inp,),config=cfg,topk=100)
        finally:hook.remove()
        self.assertEqual(calls,[1]);self.assertGreater(len(records[0]),2);self.assertLessEqual(len(records[0]),200)
        self.assertTrue(all(c.source.producer=='b0' for c in records[0]));self.assertFalse(outputs[0].yolo_only.any())
        self.assertTrue(all(not any(c.local_evidence.vector) for c in records[0]))
        loss=outputs[0].identity_logits.square().mean();loss.backward()
        grad=sum(float(p.grad.abs().sum()) for p in model.detector.backbone.parameters() if p.grad is not None);self.assertGreater(grad,0)
        # A new current dense map must produce new grid IDs; never read a saved B0 pool.
        with torch.no_grad():
            changed={k:v.detach().clone() for k,v in dense.items()};changed['center_logits'].fill_(-90);changed['center_logits'][0,0,2,2]=8;changed['quality_logits'].zero_()
            pkt,changed_records,budgets,trace=build_current_proposals(changed,features.detach(),inp,cfg,topk=1)
        self.assertEqual(len(changed_records),1);self.assertTrue(changed_records[0].candidate_id.endswith('grid:'+str(2*dense['center_logits'].shape[-1]+2)))
        baseline,_=candidate_union({k:v.detach() for k,v in dense.items()},A[0],cfg)
        baseline20,_=candidate_union({k:v.detach() for k,v in dense.items()},A[0],cfg,topk=20)
        self.assertEqual(baseline,baseline20)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6)
        optimizer.step()
        with self.assertRaisesRegex(ValueError,'Weights changed'):
            model(x,frame_inputs=(inp,),config=cfg,topk=1)
        new_source=model.record_optimizer_update(1)
        with self.assertRaisesRegex(ValueError,'Stale'):
            model(x,frame_inputs=(inp,),config=cfg,topk=1)
        self.assertEqual(new_source.optimizer_step,1)
        with torch.no_grad():
            continued=model(x,frame_inputs=(replace(inp,b0_source=new_source),),config=cfg,topk=1)
        self.assertTrue(all(c.source.optimizer_step==1 for c in continued[3][0]))
        self.assertGreater(len(continued[3][0]),0)


        # Explicit synthetic external-support fixture; no real YOLO-performance claim.
        mask=np.zeros((frame.height,frame.width),bool);mask[8:24,8:24]=True
        ysource=SourceIdentity('yolo_visual','c'*64,'d'*64,'engineering','e'*64)
        y=CandidateRecord(frame,'yolo:engine:0','raw:0',ysource,(8,8,24,24),(('score',.8),),support_kind='mask',source_mask_ref='m0',evidence_channel='visual_positive',evidence_score=.8)
        extended=replace(inp,external_candidates=(y,),supports=(PredictedSupport(frame,'m0',mask),))
        with torch.no_grad():pkt,recs,_,exttrace=build_current_proposals(changed,features.detach(),extended,cfg,topk=1)
        self.assertTrue(any(c.candidate_id==y.candidate_id for c in recs));self.assertTrue(pkt.yolo_only[-1]);self.assertEqual(len(pkt.b0_prior_ids),len(recs)-1)
        with self.assertRaises(ValueError):build_current_proposals(changed,features.detach(),replace(extended,frame=replace(frame,image_id='wrong')),cfg,topk=1)
        with self.assertRaises(ValueError):SourceIdentity('b0',registry['sha256'],'b'*64,weight_identity_kind='initialization_sha256_plus_update')
        live=SourceIdentity('b0',registry['sha256'],'b'*64,weight_identity_kind='initialization_sha256_plus_update',run_config_sha256='f'*64,optimizer_step=2)
        self.assertEqual(live.optimizer_step,2)
        (OUT/'VERIFICATION.json').write_text(json.dumps({'status':'CURRENT_DENSE_AND_SHARED_FEATURE_CONSUMPTION_PASS','image':frame.image_id,'real_B0_parent':registry['selected_step'],'single_backbone_forward':True,'candidate_trace':traces[0],'current_map_change_updates_gridIDs':True,'backbone_gradient_L1':grad,'baseline_default20_equal':True,'external_support_fixture':'synthetic only, not YOLO output','external_trace':exttrace,'wrong_frame_rejected':True,'GT_used_for_candidate_generation':False,'accuracy_gain_measured':False},indent=2)+'\n')

if __name__=='__main__':unittest.main(verbosity=2)
