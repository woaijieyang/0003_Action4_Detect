"""No-padding deterministic DDP batches with persistent-loader epoch updates."""
import numpy as np

class EpochRankBatches:
    def __init__(self,size,batch,world,rank,seed):
        if size%batch or batch%world or not 0<=rank<world:
            raise ValueError("Exact no-padding rank partition required")
        self.size,self.batch,self.world,self.rank,self.seed=size,batch,world,rank,seed
        self.set_epoch(0)
    def set_epoch(self,epoch,cursor=0):
        if cursor<0 or cursor>self.size or cursor%self.batch:
            raise ValueError("Resume cursor must be a global optimizer-batch boundary")
        self.epoch,self.cursor=int(epoch),int(cursor)
        self.order=np.random.default_rng(self.seed+self.epoch*1000003).permutation(self.size).tolist()
    def __iter__(self):
        local=self.batch//self.world
        for first in range(self.cursor,self.size,self.batch):
            start=first+self.rank*local
            yield self.order[start:start+local]
    def __len__(self):return (self.size-self.cursor)//self.batch
