"""Dataflow tests: exact rank coverage, persistent epochs and photometric geometry."""
import copy,json,torch,numpy as np
from torch.utils.data import DataLoader,Dataset
from data.action4_epoch_batches import EpochRankBatches
from data.action4_photometric import augment_prepared,DEFAULT
from data.action4_dataset import prepare_full_images
from data import DataAdapter
class IdentityData(Dataset):
    def __len__(self):return 1600
    def __getitem__(self,i):return i

def run():
    torch.set_num_threads(2)
    for epoch in [0,1,79]:
        partition=[]
        for rank in range(4):
            sampler=EpochRankBatches(1600,32,4,rank,20261023);sampler.set_epoch(epoch)
            batches=list(sampler);assert len(batches)==50 and all(len(b)==8 for b in batches)
            sampler.set_epoch(epoch,320);assert list(sampler)==batches[10:]
            partition.extend(i for batch in batches for i in batch)
        assert len(partition)==len(set(partition))==1600 and sorted(partition)==list(range(1600))
    sampler=EpochRankBatches(1600,32,4,0,20261023)
    loader=DataLoader(IdentityData(),batch_sampler=sampler,num_workers=2,persistent_workers=True)
    for epoch in [0,1]:
        sampler.set_epoch(epoch);actual=[x.tolist() for x in loader]
        assert actual==list(sampler)
    del loader
    image=torch.randint(0,256,(3,96,128),generator=torch.Generator().manual_seed(3),dtype=torch.uint8)
    x,aff=prepare_full_images([image],size=(160,128));old=x.clone();image_old=image.clone()
    cfg=copy.deepcopy(DEFAULT);cfg.update(probability=1.,operation_probability=1.)
    a,pa=augment_prepared(x,[(128,96)],[17],cfg)
    b,pb=augment_prepared(x,[(128,96)],[17],cfg)
    c,_=augment_prepared(x,[(128,96)],[18],cfg)
    assert torch.equal(a,b) and pa==pb and not torch.equal(a,c)
    assert torch.equal(x,old) and torch.equal(image,image_old) and not torch.equal(a,x)
    _,info=DataAdapter.letterbox_affine((128,96),(160,128));left,top,right,bottom=info["padding_ltrb"]
    assert top>0
    assert torch.equal(a[:,:,:top],x[:,:,:top]) and torch.equal(a[:,:,-bottom:],x[:,:,-bottom:])
    off=dict(cfg,probability=0)
    unchanged,_=augment_prepared(x,[(128,96)],[17],off);assert torch.equal(unchanged,x)
    print(json.dumps({"passed":True,"rank_coverage_1600_exact":True,"resume_suffix_exact":True,"persistent_epochs_exact":True,
        "seed_replay_exact":True,"different_seed_changes_RGB":True,"original_and_padding_unchanged":True,"scope":"engineering dataflow, not accuracy"}))
if __name__=="__main__":run()
