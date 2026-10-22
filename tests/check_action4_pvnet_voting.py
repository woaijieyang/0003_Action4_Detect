"""Occluded/outside-frame point voting and degenerate-kernel checks."""
from pathlib import Path
import sys,json,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from model.ops.pvnet_voting import ransac_voting_layer_v2,TensorVotingKernels

def main():
    torch.set_num_threads(2);torch.manual_seed(19)
    h=w=24;y,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij');pixels=torch.stack((x,y),-1).float()
    points=torch.tensor([[-4.,-6.],[33.,-3.],[32.,36.],[-7.,38.],[3.,2.],[20.,2.],[22.,20.],[2.,22.]])
    vectors=torch.nn.functional.normalize(points[None,None]-pixels[:,:,None],dim=-1)
    mask=torch.zeros(1,h,w,dtype=torch.long);mask[:,5:19,4:20]=1
    clean=ransac_voting_layer_v2(mask,vectors[None],2,64)
    noisy=vectors.clone();noise=torch.rand(h,w)<.25;noisy[noise]=torch.nn.functional.normalize(torch.randn_like(noisy[noise]),dim=-1)
    robust=ransac_voting_layer_v2(mask,noisy[None],2,64)
    ce=float(torch.linalg.vector_norm(clean[0,0]-points,dim=-1).max());re=float(torch.linalg.vector_norm(robust[0,0]-points,dim=-1).max());assert ce<.001 and re<.01
    direct=torch.zeros(10,8,2);direct[...,0]=1;coords=torch.rand(10,2);ids=torch.randint(0,10,(16,8,2));degenerate=TensorVotingKernels.generate_hypothesis(direct,coords,ids);assert torch.equal(degenerate,torch.zeros_like(degenerate))
    report={'status':'PASS','clean_max_px':ce,'outlier25pct_max_px':re,'points_outside_foreground_and_frame':True,'parallel_kernel_zero_hypothesis':True,'scope':'constructed operator evidence only, not trained accuracy; zero hypotheses must not be accepted as points'}
    out=R/'outputs/action4_a0_additive_v001';(out/'VOTING_TEST.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
