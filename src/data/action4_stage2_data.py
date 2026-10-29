"""Version-bound role units and exact, no-drop epoch partitions for independent training."""
import math
import numpy as np
import torch
from data import DataAdapter
from data.action4_component_data import ComponentData
from data.action4_online_surface_targets import prepare_online_surface_batch

class Stage2Units:
    def __init__(self,cfg,split="train",*,sam_factories=None):
        from data.action4_image_mode_contract import ensure_image_mode
        ensure_image_mode(cfg)
        from data.action4_sam_variant_selection import validate_config as validate_rgb_variants
        variant_selection=validate_rgb_variants(cfg,split)
        from data.action4_sam_roi_routing import validate_config as validate_roi_routing
        validate_roi_routing(cfg,split)
        if cfg.get("sam_mask_augmentation") is not None and (cfg.get("role")!="A0" or cfg.get("sam_enabled") is not True):
            raise ValueError("sam_mask_augmentation requires explicit SAM-enabled A0; baseline/B0 configuration refused")
        self.cfg=cfg;self.split=split
        from data.action4_prediction_roi_data import validate_config as prediction_config
        self.prediction_roi=prediction_config(cfg,split)
        variant=cfg.get("a0_variant");corner=cfg.get("corner_enabled",False)
        if (variant=="a0_corner_image" or corner is not False) and (
                cfg.get("role")!="A0" or cfg.get("sam_enabled") is not True
                or variant!="a0_corner_image" or corner is not True):
            raise ValueError("Corner training data requires explicit matching A0/SAM/corner flags")
        self.sam_provider=None;self.sam_variants=None;self.sam_factories=sam_factories
        if cfg.get("sam_enabled",False):
            if cfg.get("role")!="A0" or variant not in ("a0_sam_image","a0_corner_image") or type(corner) is not bool or corner!=(variant=="a0_corner_image") or split!="train":
                raise ValueError("Explicit SAM A0 training variant required")
            if cfg.get("photometric"):
                raise ValueError("SAM cache cannot accompany post-ROI photometric augmentation; matched native variant required")
            if sam_factories is None or len(sam_factories)!=2:
                raise ValueError("SAM training requires injected identity/MaskAux factories")
            if variant_selection is not None:
                from data.action4_sam_variant_selection import SamVariantSelector
                self.sam_variants=SamVariantSelector(cfg,split)
            else:
                from data.action4_sam_cache_provider import SamCacheProvider
                if "sam_training_associations" not in cfg:raise ValueError("SAM training association dependency missing")
                self.sam_provider=SamCacheProvider(cfg.get("sam_cache"),split=split,
                    allow_synthetic=cfg.get("mode")=="integration" and cfg.get("sam_allow_synthetic_fixture") is True,
                    training_associations=cfg["sam_training_associations"],mask_augmentation=cfg.get("sam_mask_augmentation"),roi_routing=cfg.get("sam_roi_routing"),
                    validation_workers=cfg.get("sam_cache_validation_workers",1))
        dataset_root=cfg.get("dataset_root")
        if dataset_root is None:
            self.data=ComponentData(cfg["validation_receipt"],split,cfg["role"])
        else:
            if dataset_root!="data/phrase2":raise ValueError("Unsupported explicit dataset_root; no fallback to old data")
            unpublished=bool(cfg.get("allow_unpublished_interface_validation",False))
            if unpublished and cfg.get("mode")!="integration":raise ValueError("Unpublished data only for explicit interface validation")
            from data.action4_phrase2_dataset import Phrase2Dataset
            self.data=Phrase2Dataset(split,dataset_root=dataset_root,role=cfg["role"],require_complete=not unpublished)
            if cfg["role"]=="A0":
                expected=f"data/phrase2/{split}/geometry"
                if cfg.get("surface_source",{}).get("label_roots",{}).get(split)!=expected:
                    raise ValueError("phrase2 surface labels may not use old dataset roots")
        if self.prediction_roi:
            from data.action4_prediction_roi_data import FrozenPredictionDataset
            self.data=FrozenPredictionDataset(self.data,cfg)
        selected=cfg.get(split+"_indices")
        if selected is not None and cfg["mode"]!="integration":raise ValueError("Partial data only allowed in explicit integration mode")
        self.indices=list(range(len(self.data))) if selected is None else list(map(int,selected))
        if not self.indices or len(set(self.indices))!=len(self.indices) or min(self.indices)<0 or max(self.indices)>=len(self.data):
            raise ValueError("Invalid or duplicated role units")
        if self.sam_provider is not None or self.sam_variants is not None:
            image_indices={self.data.units[i][0] for i in self.indices}
            (self.sam_variants if self.sam_variants is not None else self.sam_provider).validate_image_coverage([self.data.records[i] for i in sorted(image_indices)])
    def __len__(self):return len(self.indices)
    def __getitem__(self,unit):
        index,epoch,active,global_count=unit
        source_index=self.indices[index];item=self.data[source_index]
        provider=self.sam_provider;variant_trace=None
        if self.sam_variants is not None:
            item,provider,variant_trace=self.sam_variants.prepare_item(item,epoch=epoch)
        seed=self.cfg["seed"]+epoch*1000003+source_index*104729
        common={"data_index":source_index,"active":bool(active),"global_count":global_count,"augmentation_seed":seed}
        if self.cfg["role"]=="B0":return dict(common,item=item)
        routing_trace=None
        if self.prediction_roi:
            row=item["frozen_prediction_roi"]
            rgb,targets,g,meta=prepare_online_surface_batch([item],self.split,boxes=[row["bbox_for_a0"]],
                surface_source=self.cfg["surface_source"],roi_expansion=self.cfg["roi_expansion"],roi_source="frozen_b0_prediction")
            meta[0]["frozen_prediction_roi"]=row
            meta[0]["prediction_manifest_identity"]=item["prediction_manifest_identity"]
        elif self.cfg.get("sam_roi_routing") is not None:
            crop_spec,routing_trace=provider.training_crop_spec(item,seed=self.cfg["seed"],epoch=epoch)
            seed=crop_spec.augmentation_seed;common["augmentation_seed"]=seed
            routing_trace["active"]=bool(active)
            rgb,targets,g,meta=prepare_online_surface_batch([item],self.split,crop_specs=[crop_spec],
                surface_source=self.cfg["surface_source"])
            meta[0]["sam_roi_routing"]=routing_trace
        else:
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
        result=dict(common,roi=rgb[0],surface_masks=targets["masks"][0],surface_sparse=targets["sparse"][0],
                    geometry={k:v[0] for k,v in g.items()},metadata=meta[0])
        if variant_trace is not None:result["metadata"]["sam_rgb_variant_selection"]=variant_trace
        if provider is not None:
            result=provider.enrich_training(item,result,identity_factory=self.sam_factories[0],
                mask_aux_factory=self.sam_factories[1])
        if variant_trace is not None:
            import copy
            result["sam_input"]["dependencies"]["rgb_variant_selection"]=copy.deepcopy(variant_trace)
        return result

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
