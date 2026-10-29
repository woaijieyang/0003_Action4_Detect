"""Stage-one regressions: explicit errors/counts and real optimizer behavior."""
import copy, unittest
import numpy as np
import torch
from eval.action4_pose_protocol import score_pose,match_geometry,rank,SCHEMA
from model.training.action4_supervision import weighted_objective,step_if_supervised
from data.action4_dataset import prepare_rois
from data import DataAdapter

torch.set_num_threads(2)

def points(x=0):return [[x,0],[x+10,0],[x+10,10],[x,10]]*2
def prediction(pid='a',x=0):
    return {'prediction_id':pid,'published':True,'pose_valid':True,'positive_depth_verified':True,
            'corner_ids':list(range(8)),'corners_px':points(x)}
def record():
    return {'sample_id':'image','width':100,'height':100,'instances':[
        {'instance_id':1,'corners_px':points(),'bbox_full_xyxy':[0,0,10,10],'occlusion_inframe':.5}]}

class PoseProtocolTest(unittest.TestCase):
    def score(self,ps):return score_pose([{'sample_id':'image','instances':ps}],[record()])
    def test_perfect_duplicate_invalid_and_miss(self):
        r=self.score([prediction()]);self.assertEqual(r['pose_all8_at_002_rate'],1)
        r=self.score([prediction(),prediction('b')]);self.assertEqual((r['pose_fp'],r['pose_fn']),(1,0))
        bad=prediction();bad['positive_depth_verified']=False
        r=self.score([bad]);self.assertEqual((r['pose_fp'],r['pose_fn'],r['invalid_pose_count']),(1,1,1))
        r=self.score([]);self.assertIsNone(r['pose_corner_p95_valid_matched_only']);self.assertEqual(r['pose_fn'],1)
    def test_wrong_ids_and_corner_error(self):
        p=prediction();p['corner_ids']=list(reversed(range(8)))
        self.assertEqual(self.score([p])['invalid_pose_count'],1)
        p=prediction();p['corners_px'][0]=[1,1]
        r=self.score([p]);self.assertEqual((r['pose_fp'],r['pose_fn']),(1,1))
    def test_maximum_cardinality_not_greedy(self):
        b=[np.array([0,0,10,10]),np.array([3,0,13,10])]
        pairs=match_geometry([np.array([1,0,11,10]),np.array([-2,0,8,10])],b,['a','b'],[1,2])
        self.assertEqual(len(pairs),2)
    def test_stable_matching_and_null_rank(self):
        box=np.array([0,0,10,10])
        self.assertEqual(match_geometry([box,box],[box],['b','a'],[1]),[(1,0)])
        self.assertLess(rank(self.score([]),0),rank(self.score([prediction()]),1))

class UpdateTest(unittest.TestCase):
    def test_global_zero_after_momentum(self):
        model=torch.nn.Linear(2,1).double();opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,10)
        model(torch.ones(1,2,dtype=torch.float64)).square().sum().backward()
        self.assertTrue(step_if_supervised(model,opt,sch,{'loss':1},{})[0])
        params=copy.deepcopy(model.state_dict());state=copy.deepcopy(opt.state_dict());sched=copy.deepcopy(sch.state_dict())
        for p in model.parameters():p.grad=torch.zeros_like(p)
        self.assertFalse(step_if_supervised(model,opt,sch,{'loss':0},{})[0])
        for k,v in params.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
        for key,part in state['state'].items():
            for name,v in part.items():torch.testing.assert_close(v,opt.state_dict()['state'][key][name],rtol=0,atol=0)
        self.assertEqual(sched,sch.state_dict())
    def test_independent_head_skips_decay_and_momentum(self):
        model=torch.nn.ModuleDict({'shared':torch.nn.Linear(1,1),'quality':torch.nn.Linear(1,1)})
        opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,10)
        deps=[{'parameter_prefixes':['quality'],'losses':['quality']}]
        x=torch.ones(1,1)
        for module in model.values():module(x).square().sum().backward()
        step_if_supervised(model,opt,sch,{'main':1,'quality':1},{},deps)
        opt.zero_grad(set_to_none=True);old=copy.deepcopy(model['quality'].state_dict())
        oldstate={p:copy.deepcopy(opt.state[p]) for p in model['quality'].parameters()}
        (model['shared'](x).square().sum()+model['quality'](x).sum()*0).backward()
        step_if_supervised(model,opt,sch,{'main':1,'quality':0},{},deps)
        for k,v in old.items():torch.testing.assert_close(v,model['quality'].state_dict()[k],rtol=0,atol=0)
        for p,st in oldstate.items():
            for k,v in st.items():torch.testing.assert_close(v,opt.state[p][k],rtol=0,atol=0)
    def test_global_denominator_micro_and_rank_equivalence(self):
        # Unequal visible/hidden counts across ranks and micro-batches, including a zero dummy.
        x=torch.tensor([1.,2.,3.,4.,5.],dtype=torch.float64);y=torch.tensor([2.,1.,0.,4.,2.],dtype=torch.float64)
        masks={'visible':torch.tensor([1,1,0,0,0],dtype=torch.bool),'hidden':torch.tensor([0,0,1,1,1],dtype=torch.bool)}
        counts={k:int(m.sum()) for k,m in masks.items()};weights={'visible':1.,'hidden':.7}
        p=torch.nn.Parameter(torch.tensor(.3,dtype=torch.float64))
        whole=sum(weights[k]*((p*x-y).square()[m]).sum()/counts[k] for k,m in masks.items());whole.backward();ref=p.grad.clone()
        grads=[]
        for chunks in [[[0],[1,2]],[[3],[4],[]]]:
            q=torch.nn.Parameter(torch.tensor(.3,dtype=torch.float64))
            for indices in chunks:
                ix=torch.tensor(indices,dtype=torch.long)
                sums={k:((q*x[ix]-y[ix]).square()*m[ix]).sum() for k,m in masks.items()}
                weighted_objective(sums,counts,weights,2).backward()
            grads.append(q.grad)
        torch.testing.assert_close(sum(grads)/2,ref,rtol=1e-8,atol=1e-10)

class PixelSamplingTest(unittest.TestCase):
    def test_identity_padding_and_scale(self):
        yy,xx=torch.meshgrid(torch.arange(32),torch.arange(32),indexing='ij')
        rgb=torch.stack([xx*7,yy*7,torch.full_like(xx,128)]).to(torch.uint8)
        sampled,A=prepare_rois([rgb],[[-.5,-.5,31.5,31.5]],[0],roi_size=32,expansion=1.)
        from data.action4_dataset import _normalize
        torch.testing.assert_close(sampled[0],_normalize(rgb.float()/255),atol=1e-5,rtol=0)
        np.testing.assert_allclose(A[0],np.eye(3),atol=1e-12)
        padded,_=prepare_rois([rgb],[[-16.5,-16.5,15.5,15.5]],[0],roi_size=32,expansion=1.)
        torch.testing.assert_close(padded[0,:,0,0],_normalize(torch.zeros(3,1,1))[:,0,0],atol=1e-6,rtol=0)
        for n in (112,224,448):
            a=DataAdapter.crop_affine([-.5,-.5,447.5,447.5],(n,n))
            points=np.array([[0.,0.],[447.,447.],[123.25,241.5]])
            np.testing.assert_allclose(DataAdapter.transform_points(DataAdapter.transform_points(points,a),np.linalg.inv(a)),points,atol=1e-3)

if __name__=='__main__':unittest.main()
