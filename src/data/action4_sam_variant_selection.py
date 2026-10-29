"""Explicit integration-only image-level native RGB selection before any ROI crop.

This module loads existing verified cache leaves. It never generates RGB variants,
SAM masks or model predictions, and never changes original geometry annotations.
"""
import copy
import math
import numpy as np
from .action4_sam_native_export import canonical,_v2_checked,_v2_rgb,COMPLETED_MODE,VARIANT_MODE
from .action4_sam_cache_provider import SamCacheProvider

SCHEMA='action4.sam_rgb_variant_selection.v1'
NAMES=('original','light','blur')


def _weights(probabilities):
    if not isinstance(probabilities,dict) or not probabilities or not set(probabilities)<=set(NAMES):
        raise ValueError('Explicit nonempty original/light/blur probabilities required')
    if any(type(p) not in (int,float) or not math.isfinite(p) or p<=0 for p in probabilities.values()):
        raise ValueError('Every declared leaf needs a strictly positive finite probability')
    if not math.isclose(math.fsum(probabilities.values()),1.,rel_tol=0.,abs_tol=1e-12):
        raise ValueError('Declared probabilities must already be normalized to one')
    return {name:float(probabilities[name]) for name in sorted(probabilities)}


def choose_leaf(probabilities,*,seed,epoch,image_id,selection_rule="integration_weighted_v1"):
    """Stateless draw keyed only by base seed, epoch and original image identity.

    Lexically ordered leaves make declaration order immaterial. No instance,
    rank, worker, input order, prediction availability or quality enters the RNG.
    """
    weights=_weights(probabilities)
    if type(seed) is not int or seed<0 or type(epoch) is not int or epoch<0:
        raise ValueError('Nonnegative integer base seed and epoch required')
    if not isinstance(image_id,str) or not image_id.strip():raise ValueError('Original image ID required')
    if selection_rule == "native_policy_one_based_v1":
        if set(weights) != set(NAMES) or any(abs(v-1/3)>1e-12 for v in weights.values()):
            raise ValueError("Plan selector requires all three equally weighted leaves")
        from .action4_rgb_variants import select_variant,DEFAULT_POLICY
        epoch_one_based=epoch+1
        selected=select_variant(image_id,epoch_one_based,seed=seed,policy=DEFAULT_POLICY)
        identity=canonical(dict(rule=selection_rule,seed=seed,epoch_one_based=epoch_one_based,image_id=image_id,policy=DEFAULT_POLICY))
        return dict(base_seed=seed,epoch=epoch,epoch_one_based=epoch_one_based,image_id=image_id,
            draw_identity=identity,draw=None,probabilities=weights,selected_leaf=selected,selection_rule=selection_rule)
    if selection_rule != "integration_weighted_v1":raise ValueError("Unknown RGB selection rule")
    identity=canonical({'schema':SCHEMA,'base_seed':seed,'epoch':epoch,'image_id':image_id})
    # Exactly representable 53-bit uniform value in [0,1); no RNG state to resume.
    draw=(int(identity[:16],16)>>11)/float(1<<53)
    cumulative=0.;selected=next(reversed(weights))
    for name,weight in weights.items():
        cumulative+=weight
        if draw<cumulative:
            selected=name
            break
    return {'base_seed':seed,'epoch':epoch,'image_id':image_id,'draw_identity':identity,
        'draw':draw,'probabilities':weights,'selected_leaf':selected}


def validate_config(cfg,split='train'):
    if 'sam_rgb_variants' not in cfg:return None
    if (cfg.get('mode')!='integration' or split!='train' or cfg.get('role')!='A0'
            or cfg.get('sam_enabled') is not True or cfg.get('a0_variant') not in ('a0_sam_image','a0_corner_image')):
        raise ValueError('RGB variant selection is explicit SAM A0 integration/train only; no formal/eval/B0 admission')
    if cfg.get('photometric') not in (None,False):
        raise ValueError('Variant selection cannot combine post-ROI photometric changes')
    if cfg.get('sam_roi_routing') is not None:
        from .action4_sam_roi_routing import validate_policy as routing_policy
        routing_policy(_v2_checked(cfg['sam_roi_routing']))
    if cfg.get('sam_mask_augmentation') is not None:
        from .action4_sam_mask_augmentation import validate_policy
        validate_policy(_v2_checked(cfg['sam_mask_augmentation']))
    if cfg.get('sam_cache') is not None or cfg.get('sam_training_associations') is not None:
        raise ValueError('Variant leaves cannot coexist with a global single-cache/association source')
    if cfg.get('sam_allow_synthetic_fixture'):
        raise ValueError('Variant selection requires verified native leaves, not synthetic fixtures')
    if not isinstance(cfg.get('train_indices'),list) or not cfg['train_indices']:
        raise ValueError('Integration variant selection requires explicitly selected train indices')
    if type(cfg.get('seed')) is not int or cfg['seed']<0:raise ValueError('Explicit nonnegative base seed required')
    spec=cfg['sam_rgb_variants']
    if not isinstance(spec,dict) or not {'schema','leaves'}<=set(spec) or set(spec)-{'schema','leaves','selection_rule'} or spec['schema']!=SCHEMA:
        raise ValueError('Explicit RGB selection schema/leaves required')
    leaves=spec['leaves']
    if not isinstance(leaves,list) or not leaves:raise ValueError('Nonempty declared leaves required')
    names=[]
    for leaf in leaves:
        if not isinstance(leaf,dict) or set(leaf) not in (
                {'name','probability','sam_cache','sam_training_associations'},
                {'name','probability','shards'}):
            raise ValueError('Each leaf needs a name, probability and explicit cache/association shards')
        if not isinstance(leaf['name'],str) or leaf['name'] not in NAMES:raise ValueError('Unknown native RGB leaf name')
        if 'shards' in leaf:
            if not isinstance(leaf['shards'],list) or not leaf['shards']:raise ValueError('Nonempty cache shards required')
            names.append(leaf['name'])
            continue
        cache=leaf['sam_cache']
        if not isinstance(cache,dict) or set(cache)!={'manifest','expected_sources'}:
            raise ValueError('Explicit leaf cache manifest and expected sources required')
        for binding in (cache['manifest'],leaf['sam_training_associations']):
            if not isinstance(binding,dict) or set(binding)!={'path','sha256'}:
                raise ValueError('Explicit leaf path/SHA binding required')
        names.append(leaf['name'])
    if len(set(names))!=len(names):raise ValueError('Duplicate RGB selection leaf')
    probabilities=_weights({leaf['name']:leaf['probability'] for leaf in leaves})
    choose_leaf(probabilities,seed=cfg['seed'],epoch=0,image_id='configuration-validation',
        selection_rule=spec.get('selection_rule','integration_weighted_v1'))
    return copy.deepcopy(spec)


class SamVariantSelector:
    def __init__(self,cfg,split='train'):
        spec=validate_config(cfg,split)
        if spec is None:raise ValueError('Explicit sam_rgb_variants configuration required')
        self.spec=spec;self.base_seed=cfg['seed']
        self.leaves={leaf['name']:leaf for leaf in spec['leaves']}
        self.probabilities=_weights({name:leaf['probability'] for name,leaf in self.leaves.items()})
        self.selection_identity=canonical({'schema':SCHEMA,'leaves':[self.leaves[name] for name in sorted(self.leaves)],'selection_rule':spec.get('selection_rule','integration_weighted_v1')})
        self.providers={}
        for name,leaf in self.leaves.items():
            if 'shards' in leaf:
                from .action4_sam_cache_collection import SamCacheCollection
                self.providers[name]=SamCacheCollection(name,leaf['shards'],split=split,mask_augmentation=cfg.get('sam_mask_augmentation'),roi_routing=cfg.get('sam_roi_routing'))
                continue
            index=_v2_checked(leaf['sam_cache']['manifest'])
            expected_mode=COMPLETED_MODE if name=='original' else VARIANT_MODE
            if (index.get('sources',{}).get('upstream_mode')!=expected_mode
                    or index.get('rgb_variant')!=('native_original' if name=='original' else name)):
                raise ValueError('Declared leaf differs from verified native cache RGB variant')
            provider=SamCacheProvider(leaf['sam_cache'],split=split,training_associations=leaf['sam_training_associations'],mask_augmentation=cfg.get('sam_mask_augmentation'),roi_routing=cfg.get('sam_roi_routing'))
            if not (provider.production_associations if name=='original' else provider.variant_associations):
                raise ValueError('Leaf requires its corresponding verified supervision ownership')
            self.providers[name]=provider

    def validate_image_coverage(self,records):
        # Every declared leaf, regardless of draw or mask status, must cover all selected images.
        for provider in self.providers.values():provider.validate_image_coverage(records)

    def prepare_item(self,item,*,epoch):
        import torch
        record=item['record'];image_id=record['sample_id']
        decision=choose_leaf(self.probabilities,seed=self.base_seed,epoch=epoch,image_id=image_id,selection_rule=self.spec.get('selection_rule','integration_weighted_v1'))
        name=decision['selected_leaf'];provider=self.providers[name];leaf=self.leaves[name]
        if 'shards' in leaf:
            collection=provider
            provider=collection.provider_for(image_id)
            leaf=leaf['shards'][collection.owners[image_id]]
        provider.validate_image_coverage([record])
        entry=provider.images[image_id]
        rgb=_v2_rgb(entry['image_path'],entry['source_sha256'])
        frame,_=provider.frame_predictions(image_id,rgb)
        original=entry.get('original_source',{'path':entry['image_path'],'sha256':entry['source_sha256']})
        trace=dict(decision,schema=SCHEMA,selection_identity=self.selection_identity,
            variant_id=frame.variant_id,actual_input={'path':entry['image_path'],'sha256':entry['source_sha256']},
            variant_pixels_sha256=frame.variant_sha256,original_source=copy.deepcopy(original),
            cache_manifest=copy.deepcopy(leaf['sam_cache']['manifest']),
            training_associations=copy.deepcopy(leaf['sam_training_associations']),
            formal_training_admitted=False)
        trace['choice_identity']=canonical(trace)
        # Keep the original annotation object and geometry. Only native RGB changes.
        selected=dict(item,image=torch.from_numpy(np.array(frame.pixels,copy=True)).permute(2,0,1).contiguous())
        return selected,provider,trace
