import copy,json,os,unittest
from pathlib import Path
from dataclasses import replace
import torch
from model.action4_b0_fusion import GuidedB0,GuidanceProposals,PROTOCOL
from model.action4_local_evidence import LocalEvidenceBatch
from model.action4_guidance_contract import LocalEvidence,EvidenceMatch
from model.action4_b0_guidance_heads import GuidanceTargets,guidance_loss_sums,POSITIVE,BACKGROUND
from model.action4_release_bundle import load_bundle,warm_start
from model.training.action4_stage2_core import construct,optimizer_for,bundle_snapshot,losses_on_batch,validate_run_gate
from model.training.action4_supervision import weighted_objective,step_if_supervised

ROOT=Path('/data/zy/vln/a-test');OUT=ROOT/'outputs/action4_b0_fusion_wiring_v001'
class GuidedB0WiringTests(unittest.TestCase):
    def test_real_parent_complete_model_updates_and_strict_bundle(self):
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'),'')
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(602)
        baseline,_,parent=load_bundle(str(ROOT/'models/default/B0.pt'),'B0','cpu')
        cfg=copy.deepcopy(parent['config']);cfg.update(role='B0',mode='integration',parent_bundle=str(ROOT/'models/default/B0.pt'),parent_best_step=parent['selected_step'],b0_variant='b0_fusion_image',guidance_lr=1e-4,evaluation_scoring_policy='center')
        model,initial=construct(cfg,'cpu');self.assertIsInstance(model,GuidedB0)
        for k,v in baseline.state_dict().items():self.assertTrue(torch.equal(model.state_dict()[k],v))
        opt=optimizer_for(model,cfg);owned=[id(p) for g in opt.param_groups for p in g['params']]
        self.assertEqual(len(owned),len(set(owned)));self.assertEqual(set(owned),{id(p) for p in model.parameters() if p.requires_grad})
        ids=('b0:test:grid:1','yolo:test:2');e=LocalEvidence(text_positive=EvidenceMatch(.8,True,'matched','unique',('source',)))
        evidence=LocalEvidenceBatch(ids,(e,e),())
        boxes=torch.tensor([[15.,20,60,70],[65,35,110,100]])
        pkt=GuidanceProposals('engineering-image',ids,boxes,torch.eye(3),evidence,torch.tensor([False,True]),torch.tensor([[.8,.7,.56]]),(ids[0],))
        images=torch.rand(1,3,128,128)
        labels=GuidanceTargets(ids,torch.tensor([BACKGROUND,POSITIVE]),torch.tensor([False,True]),torch.tensor([False,True]),torch.tensor([[float('nan')]*4,[60.,30,115,105]]))
        model.eval();records=[]
        initial_guidance={k:v.detach().clone() for k,v in model.guidance.state_dict().items()}
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            dense,features,outputs=model(images,proposal_packets=(pkt,),canvas_size=(128,128),image_ids=('engineering-image',))
            losses=guidance_loss_sums(outputs[0],labels);objective=weighted_objective(losses.sums,losses.counts,{},1);objective.backward()
            grad=sum(float(p.grad.abs().sum()) for p in model.detector.backbone.parameters() if p.grad is not None)
            self.assertGreater(grad,0);self.assertGreater(float(model.guidance.identity_head.weight.grad.abs().sum()),0)
            did,norm=step_if_supervised(model,opt,torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1),losses.counts,{},model.independent_dependencies())
            self.assertTrue(did);records.append({'update':step+1,'objective':float(objective.detach()),'backbone_grad_L1':grad,'norm':norm})
        self.assertTrue(any(not torch.equal(v,initial_guidance[k]) for k,v in model.guidance.state_dict().items()))
        with torch.no_grad():expected=model(images,proposal_packets=(pkt,),canvas_size=(128,128),image_ids=('engineering-image',))[2][0]
        state=bundle_snapshot(model,initial,cfg,2);path=OUT/'engineering_B0_guidance_step2.pt';torch.save(state,path)
        reloaded,_,loaded=load_bundle(str(path),'B0','cpu');self.assertEqual(loaded['guidance_protocol'],PROTOCOL)
        with torch.no_grad():actual=reloaded(images,proposal_packets=(pkt,),canvas_size=(128,128),image_ids=('engineering-image',))[2][0]
        for key in ['identity_logits','full_boxes','quality_logits']:torch.testing.assert_close(getattr(expected,key),getattr(actual,key),rtol=0,atol=0)
        with self.assertRaises(ValueError):reloaded(images)
        with self.assertRaises(ValueError):reloaded(images,proposal_packets=(replace(pkt,image_id='other'),),canvas_size=(128,128),image_ids=('engineering-image',))
        with self.assertRaises(ValueError):losses_on_batch(model,[],cfg,'cpu')
        with self.assertRaises(ValueError):warm_start(str(path),'B0','cpu')
        with self.assertRaises(ValueError):validate_run_gate(cfg,{'mode':'integration','launch_allowed':True})
        bad=copy.deepcopy(state);bad['model'].pop('guidance.identity_head.weight');badpath=OUT/'missing_guidance_parameter.pt';torch.save(bad,badpath)
        with self.assertRaises(RuntimeError):load_bundle(str(badpath),'B0','cpu')
        (OUT/'VERIFICATION.json').write_text(json.dumps({'status':'COMPONENT_CONSTRUCT_OPTIMIZER_GRADIENT_BUNDLE_PASS','actual_parent_B0':parent['selected_step'],'fixture':'synthetic128RGB and explicit typed candidate packets; not actual predicted candidate provider','updates':records,'strict_reload_bit_exact':True,'optimizer_all_parameters_once':True,'wrong_image_and_legacy_paths_rejected':True,'formal_training_allowed':False,'remaining':'current predicted candidate generation, globalcounts trainer and guided async evaluation','accuracy_gain':False,'CUDA_initialized':torch.cuda.is_initialized()},indent=2)+'\n')

if __name__=='__main__':unittest.main(verbosity=2)
