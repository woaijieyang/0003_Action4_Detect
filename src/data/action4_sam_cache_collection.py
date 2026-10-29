"""Identity-preserving SAM cache shard routing; integration only via the caller.

Index admission is metadata-only. The selected shard's existing full provider
validates native provenance and supervision before any mask is returned.
"""
from collections import OrderedDict
import copy
from .action4_sam_native_export import _v2_checked,COMPLETED_MODE,VARIANT_MODE
from .action4_sam_cache_provider import SamCacheProvider

class SamCacheCollection:
    def __init__(self,name,shards,*,split="train",mask_augmentation=None,roi_routing=None):
        if name not in ("original","light","blur") or split!="train":
            raise ValueError("Explicit train RGB leaf required")
        if not isinstance(shards,list) or not shards:
            raise ValueError("Nonempty shard bindings required")
        self.name=name;self.split=split;self.shards=copy.deepcopy(shards)
        self.mask_augmentation=mask_augmentation;self.roi_routing=roi_routing;self.images={};self.owners={}
        self.providers=OrderedDict();self.hot_providers=OrderedDict()
        for i,shard in enumerate(self.shards):
            if not isinstance(shard,dict) or set(shard)!={"sam_cache","sam_training_associations"}:
                raise ValueError("Each shard needs cache and supervision bindings")
            spec=shard["sam_cache"]
            if not isinstance(spec,dict) or set(spec)!={"manifest","expected_sources"}:
                raise ValueError("Explicit shard cache identity required")
            for b in (spec["manifest"],shard["sam_training_associations"]):
                if not isinstance(b,dict) or set(b)!={"path","sha256"}:
                    raise ValueError("Explicit shard file binding required")
            index=_v2_checked(spec["manifest"])
            if (index.get("split")!=split or index.get("origin")!="predicted_sam"
                    or index.get("prompt_origin")!="frozen_prediction_only"
                    or index.get("sources")!=spec["expected_sources"]
                    or index.get("rgb_variant")!=("native_original" if name=="original" else name)
                    or index["sources"].get("upstream_mode")!=(COMPLETED_MODE if name=="original" else VARIANT_MODE)):
                raise ValueError("Shard source or RGB leaf mismatch")
            if not index.get("images"):raise ValueError("Empty cache shard")
            for entry in index["images"]:
                sid=entry["image_id"]
                if sid in self.images:raise ValueError("Duplicate image ownership across shards")
                self.images[sid]=entry;self.owners[sid]=i
    def validate_image_coverage(self,records):
        # No native-mask quality or inference success is inferred from this registry.
        for record in records:
            sid=record["sample_id"]
            if sid not in self.images:raise ValueError("Selected image absent from declared shard domain")
            entry=self.images[sid]
            original=entry.get("original_source",{"path":entry["image_path"],"sha256":entry["source_sha256"]})
            if original!={"path":record["image"],"sha256":record["rgb_sha256"]}:
                raise ValueError("Shard original annotation image identity mismatch")
    def provider_for(self,image_id):
        if image_id not in self.owners:raise ValueError("No shard for requested image")
        i=self.owners[image_id]
        if i not in self.providers:
            shard=self.shards[i]
            provider=SamCacheProvider(shard["sam_cache"],split=self.split,
                training_associations=shard["sam_training_associations"],mask_augmentation=self.mask_augmentation,roi_routing=self.roi_routing)
            if not (provider.production_associations if self.name=="original" else provider.variant_associations):
                raise ValueError("Selected shard supervision not qualified")
            self.providers[i]=provider
        # Keep lightweight verified ownership metadata across all declared shards;
        # evict decoded frames only, rather than reconstructing the full provider.
        from .action4_sam_native_export import _v2_sha
        shard=self.shards[i]
        for binding in (shard["sam_cache"]["manifest"],shard["sam_training_associations"]):
            if _v2_sha(binding["path"])!=binding["sha256"]:raise ValueError("Selected shard binding changed")
        self.hot_providers[i]=None;self.hot_providers.move_to_end(i)
        while len(self.hot_providers)>2:
            cold,_=self.hot_providers.popitem(last=False)
            self.providers[cold].frames.clear()
        return self.providers[i]
