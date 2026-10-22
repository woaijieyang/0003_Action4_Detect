"""PVNet official v2 voting loop with documented PyTorch compatibility changes.

Upstream zju3dv/pvnet commit e5c74f73c31f2b3f50bb10f02292a4dcb01e649a.
Apache-2.0, see PVNET_LICENSE. Original Python voting/refinement loop retained.
Legacy CUDA kernels are expressed as tensor operations; no toolkit installation.
Changed: explicit devices, boolean indexing, re-sample pairs each round rather
than accumulating confidence on the same repeated hypotheses. Not a full PVNet.
"""
import torch
import numpy as np


def log_msg(message):
    pass


class TensorVotingKernels:
    @staticmethod
    def generate_hypothesis(direct, coords, indices):
        indices=indices.long()
        keypoint=torch.arange(direct.shape[1],device=direct.device)[None]
        v0=direct[indices[...,0],keypoint]
        v1=direct[indices[...,1],keypoint]
        c0=coords[indices[...,0]]
        c1=coords[indices[...,1]]
        n0=torch.stack((v0[...,1],-v0[...,0]),dim=-1)
        n1=torch.stack((v1[...,1],-v1[...,0]),dim=-1)
        rhs0=(n0*c0).sum(-1)
        rhs1=(n1*c1).sum(-1)
        denominator=n0[...,0]*n1[...,1]-n0[...,1]*n1[...,0]
        valid=denominator.abs()>=1e-6
        safe=torch.where(valid,denominator,torch.ones_like(denominator))
        x=(rhs0*n1[...,1]-n0[...,1]*rhs1)/safe
        y=(n0[...,0]*rhs1-rhs0*n1[...,0])/safe
        return torch.where(valid[...,None],torch.stack((x,y),-1),torch.zeros_like(c0))

    @staticmethod
    def voting_for_hypothesis(direct,coords,hypotheses,inliers,threshold):
        directions=direct.permute(1,0,2)[None]
        norm1=torch.linalg.vector_norm(directions,dim=-1)
        for start in range(0,len(hypotheses),32):
            delta=hypotheses[start:start+32,:,None]-coords[None,None]
            norm2=torch.linalg.vector_norm(delta,dim=-1)
            cosine=(delta*directions).sum(-1)/(norm1*norm2).clamp_min(1e-12)
            votes=(cosine>threshold)&(norm1>=1e-6)&(norm2>=1e-6)
            inliers[start:start+32].copy_(votes.to(inliers.dtype))


ransac_voting=TensorVotingKernels()

def ransac_voting_layer_v2(mask, vertex, class_num, round_hyp_num, inlier_thresh=0.999, confidence=0.99, max_iter=20,
                           min_num=5,max_num=30000,refine_iter_num=1):
    '''
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,vn,2]
    :param class_num:
    :param round_hyp_num:
    :param inlier_thresh:
    :return: [b,cn,vn,2]
    '''

    log_msg('ransac begin')
    b,h,w,vn,_=vertex.shape
    batch_win_pts=[]
    for bi in range(b):
        class_win_pts = []
        hyp_num=0
        for k in range(class_num-1):
            cur_mask=mask[bi]==k+1
            foreground=torch.sum(cur_mask)
            log_msg('get sum')

            # if too few points, just skip it
            if foreground<min_num:
                all_win_pts=torch.zeros([vn,2],dtype=torch.float32,device=mask.device)
                class_win_pts.append(torch.unsqueeze(all_win_pts,0)) # [1,vn,2]
                continue

            # if too many inliers, we randomly down sample it
            if foreground>max_num:
                selection=torch.zeros(cur_mask.shape,dtype=torch.float32,device=mask.device).uniform_(0,1)
                selected_mask=(selection<(max_num/foreground.float()))
                cur_mask*=selected_mask

            log_msg('test done')
            coords=torch.nonzero(cur_mask).float()   # [tn,2]
            coords=coords[:,[1,0]]
            log_msg('nonzero')
            direct=vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask,2),3)) # [tn,vn,2]
            direct=direct.view([coords.shape[0],vn,2])
            log_msg('mask select')
            tn=coords.shape[0]
            idxs=torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
            log_msg('random sample')

            all_win_ratio=torch.zeros([vn],dtype=torch.float32,device=mask.device)
            all_win_pts=torch.zeros([vn,2],dtype=torch.float32,device=mask.device)
            log_msg('zeros')

            cur_iter=0
            while True:
                idxs.random_(0, direct.shape[0])  # new independent hypotheses each round
                # generate hypothesis
                cur_hyp_pts=ransac_voting.generate_hypothesis(direct, coords, idxs) # [hn,vn,2]
                log_msg('generate_hypothesis')

                # voting for hypothesis
                cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
                ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh) # [hn,vn,tn]
                log_msg('voting_for_hypothesis')

                # find max
                cur_inlier_counts=torch.sum(cur_inlier,2)    # [hn,vn]
                cur_win_counts,cur_win_idx=torch.max(cur_inlier_counts,0) # [vn]
                cur_win_pts=cur_hyp_pts[cur_win_idx, torch.arange(vn,device=mask.device)]
                cur_win_ratio=cur_win_counts.float()/tn
                log_msg('find max')


                larger_mask=all_win_ratio<cur_win_ratio
                all_win_pts[larger_mask,:]=cur_win_pts[larger_mask,:]
                all_win_ratio[larger_mask]=cur_win_ratio[larger_mask]
                log_msg('mask larger')

                hyp_num+=round_hyp_num
                cur_iter+=1
                cur_min_ratio=torch.min(all_win_ratio)
                log_msg('check condition')
                if (1-(1-cur_min_ratio**2)**hyp_num)>confidence or cur_iter>max_iter:
                    break

            normal=torch.zeros_like(direct)
            normal[:,:,0]=direct[:,:,1]
            normal[:,:,1]=-direct[:,:,0]
            # compute mean intersection again
            for k in range(refine_iter_num):
                all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
                all_win_pts=torch.unsqueeze(all_win_pts,0) # [1,vn,2]
                ransac_voting.voting_for_hypothesis(direct, coords, all_win_pts, all_inlier, inlier_thresh)  # [1,vn,tn]


                log_msg('refine voting')
                refine_pts=[]
                for vi in range(vn):
                    cur_coords=coords[all_inlier[0,vi].bool()] # in,2
                    if cur_coords.shape[0]==0:
                        refine_pts.append(torch.zeros([1,2], device=mask.device))
                        continue
                    cur_normal=normal[:,vi,:][all_inlier[0,vi].bool()] # in,2

                    A=cur_normal                                   # [cn,2]
                    b=torch.sum(cur_normal*cur_coords,1)           # [cn]
                    refine_pt=torch.matmul(torch.pinverse(A),b)    # [2]
                    refine_pts.append(torch.unsqueeze(refine_pt,0))
                    log_msg('invers ')

                refine_pts=torch.cat(refine_pts,0)
                all_win_pts=refine_pts

            class_win_pts.append(torch.unsqueeze(all_win_pts,0)) # [1,vn,2]

        batch_win_pts.append(torch.unsqueeze(torch.cat(class_win_pts,0),0)) # [1,cn,vn,2]
        log_msg('class append')

    batch_win_pts=torch.cat(batch_win_pts,0)
    log_msg('batch append')

    return batch_win_pts
