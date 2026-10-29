"""Independent policy for both targets in the additional 30%-75% batch."""
import math

INTERVALS=((.30,.39),(.39,.48),(.48,.57),(.57,.66),(.66,.75))

def occlusion_bin(value):
    if value is None or not math.isfinite(value) or value<.30 or value>.75:return None
    for i,(lo,hi) in enumerate(INTERVALS):
        if (value>=lo if i==0 else value>lo) and value<=hi:return i
    raise ValueError('Finite in-range occlusion did not map to a fixed interval')

def assert_configuration(cfg):
    if cfg['batch_id']!='occlusion500_v001' or cfg['dataset_root']!='data/phrase2':raise ValueError('Wrong batch identity')
    if tuple(map(tuple,cfg['occlusion_intervals']))!=INTERVALS:raise ValueError('Frozen additional occlusion policy changed')
    if cfg['split_counts']!={'train':400,'eval':50,'test':50}:raise ValueError('Additional counts changed')
    if cfg['candidate_root']!='data/phrase2/.staging_batches/occlusion500_v001':raise ValueError('Candidate isolation changed')
    if cfg['master_seed']==2026102302:raise ValueError('Additional batch must have a separate random stream')

def source_eligible(occlusion,primary):
    return primary=='own_wrist_forearm' or (occlusion<=.35 and primary=='own_mount')

def eligible_sources(sources,bins):
    for side,m in sources.items():
        o=m['estimated_external_occlusion'];kind=bins[side];lo,hi=INTERVALS[kind]
        if not math.isfinite(o) or o<max(0,lo-.04) or o>min(1,hi+.04):return False,'scene_occlusion_estimate'
        if not source_eligible(o,m['primary_source_class']):return False,'additional_batch_source_rule'
        counts=m['occluder_class_counts'];total=sum(counts.values())
        if total>=5 and counts.get('environment',0)+counts.get('other_actor',0)>.5*total:return False,'non_own_arm_primary_occlusion'
    return True,None
