"""Keep the original strict precision bound; test the added robust stage separately."""
from pathlib import Path
import sys,json,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from model.ops.pvnet_voting import ransac_voting_layer_v2
from model.ops.action4_vote_refinement import refine_votes

def main():
    torch.set_num_threads(2);h=w=24;y,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij');pixels=torch.stack((x,y),-1).float()
    points=torch.tensor([[-4.,-6.],[33.,-3.],[32.,36.],[-7.,38.],[3.,2.],[20.,2.],[22.,20.],[2.,22.]])
    vectors=torch.nn.functional.normalize(points[None,None]-pixels[:,:,None],dim=-1);mask=torch.zeros(1,h,w,dtype=torch.long);mask[:,5:19,4:20]=1;rows=[]
    for seed in (19,31,47,83,101):
        torch.manual_seed(seed)
        clean=ransac_voting_layer_v2(mask,vectors[None],2,64)
        noisy=vectors.clone();noise=torch.rand(h,w)<.25;noisy[noise]=torch.nn.functional.normalize(torch.randn_like(noisy[noise]),dim=-1)
        original=ransac_voting_layer_v2(mask,noisy[None],2,64)[:,0]
        result=refine_votes(mask,noisy[None],original)
        raw_error=float(torch.linalg.vector_norm(original[0]-points,dim=-1).max());error=float(torch.linalg.vector_norm(result['points'][0]-points,dim=-1).max())
        assert result['valid'].all() and error<.01,(seed,error)
        assert torch.linalg.vector_norm(clean[0,0]-points,dim=-1).max()<.001
        rows.append(dict(seed=seed,original_max_px=raw_error,refined_max_px=error))
    parallel=torch.zeros_like(vectors)[None];parallel[...,0]=1
    bad=refine_votes(mask,parallel,points[None]);assert not bad['valid'].any()
    empty=refine_votes(torch.zeros_like(mask),vectors[None],points[None]);assert not empty['valid'].any()
    from model.action4_a0_enhancement import auxiliary_vote_evidence
    logits=torch.zeros(1,8,h,w)
    fg=torch.zeros(1,2,h,w);fg[:,0]=1
    empty_output={'corner_logits':logits,'direction_vectors':vectors.permute(2,3,0,1)[None],'foreground_logits':fg}
    empty_evidence=auxiliary_vote_evidence(empty_output)
    assert not empty_evidence['numerically_valid'].any()
    assert torch.equal(empty_evidence['base_corners_hm'],empty_evidence['auxiliary_corners_hm'])
    assert empty_evidence['production_replacement_enabled'] is False
    report=dict(status='PASS',cases=rows,parallel_rejected=True,empty_rejected=True,original_failure_retained=True,scope='constructed robustness only; no learned accuracy or deployment claim')
    (R/'outputs/action4_a0_additive_v001/REFINEMENT_TEST.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
