"""Version-bound role units and exact, no-drop epoch partitions for independent training."""
import math
import numpy as np
import torch
from data import DataAdapter
from data.action4_component_data import ComponentData
from data.action4_online_surface_targets import prepare_online_surface_batch

class Stage2Units:
    def __init__(self,cfg,split="train"):
        self.cfg=cfg;self.split=split
        self.data=ComponentData(cfg["validation_receipt"],split,cfg["role"])
        selected=cfg.get(split+"_indices")
        if selected is not None and cfg["mode"]!="integration":raise ValueError("Partial data only allowed in explicit integration mode")
        self.indices=list(range(len(self.data))) if selected is None else list(map(int,selected))
        if not self.indices or len(set(self.indices))!=len(self.indices) or min(self.indices)<0 or max(self.indices)>=len(self.data):
            raise ValueError("Invalid or duplicated role units")
    def __len__(self):return len(self.indices)
    def __getitem__(self,unit):
        index,epoch,active,global_count=unit
        source_index=self.indices[index];item=self.data[source_index]
        seed=self.cfg["seed"]+epoch*1000003+source_index*104729
        common={"data_index":source_index,"active":bool(active),"global_count":global_count,"augmentation_seed":seed}
        if self.cfg["role"]=="B0":return dict(common,item=item)
        rgb,targets,g,meta=prepare_online_surface_batch([item],self.split,seeds=[seed],
            jitter_config=self.cfg["roi_jitter"],surface_source=self.cfg["surface_source"])
        if self.cfg.get("photometric"):
            from data.action4_photometric import augment_prepared
            augmented,parameters=augment_prepared(rgb,[(448,448)],[seed],self.cfg["photometric"])
            up,_=DataAdapter.letterbox_affine((112,112),(448,448));A=up@np.asarray(meta[0]["affine_hm"])
            if abs(A[0,1])+abs(A[1,0])>1e-12:raise ValueError("Current ROI padding adapter assumes axis-aligned crop")
            x=(np.arange(448)-A[0,2])/A[0,0];y=(np.arange(448)-A[1,2])/A[1,1]
            record=item["record"];inside=(y[:,None]>=0)&(y[:,None]<=record["height"]-1)&(x[None,:]>=0)&(x[None,:]<=record["width"]-1)
            rgb=torch.where(torch.from_numpy(inside)[None,None],augmented,rgb)
            meta[0]["photometric_parameters"]=parameters[0]
        return dict(common,roi=rgb[0],surface_masks=targets["masks"][0],surface_sparse=targets["sparse"][0],
                    geometry={k:v[0] for k,v in g.items()},metadata=meta[0])

class Stage2Batches:
    def __init__(self,size,batch,world,rank,seed):
        if min(size,batch,world)<=0 or not 0<=rank<world or batch<world:raise ValueError("Invalid exact batch partition")
        self.size,self.batch,self.world,self.rank,self.seed=size,batch,world,rank,seed
        self.set_epoch(0)
    def set_epoch(self,epoch,cursor=0):
        if cursor<0 or cursor>self.size or (cursor!=self.size and cursor%self.batch):raise ValueError("Resume must use global update boundary")
        self.epoch,self.cursor=int(epoch),int(cursor)
        self.order=np.random.default_rng(self.seed+self.epoch*1000003).permutation(self.size)
    def __len__(self):return math.ceil((self.size-self.cursor)/self.batch)
    def __iter__(self):
        for first in range(self.cursor,self.size,self.batch):
            units=self.order[first:first+self.batch];local=np.array_split(units,self.world)[self.rank]
            # Empty rank executes a zero-weight dummy to maintain DDP collectives.
            # Dummy never contributes to losses, target counts, epoch coverage or metrics.
            yield [(int(i),self.epoch,True,len(units)) for i in local] if len(local) else [(int(units[0]),self.epoch,False,len(units))]
