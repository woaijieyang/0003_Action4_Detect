"""Explicit selected-model ROI contracts for saved corner videos.
No model or checkpoint is selected here; completed seals remain required.
"""
SUPPORTED=(1.5,1.15)

def ratio_tag(ratio):
    if type(ratio) not in (int,float) or ratio not in SUPPORTED:raise ValueError("Unregistered selected video ROI")
    return 'e'+format(ratio,'.2f').replace('.','p')

def config_ratio(cfg):
    values=cfg.get('ratios')
    if not isinstance(values,list) or len(values)!=1:raise ValueError('One selected video ROI required')
    value=values[0];ratio_tag(value)
    if value==1.15 and not isinstance(cfg.get('corner_trial_binding'),dict):raise ValueError('ROI1.15 requires completed selected corner trial binding')
    return value

def trial_ratio(selection,seal=None):
    value=selection.get('roi_expansion');ratio_tag(value);status=selection.get('status')
    if status=='PRESERVED_ALL8_DEVELOPMENT_CANDIDATE_NOT_FINAL_SEAL':
        if value!=1.5 or seal is not None:raise ValueError('Historical provisional trial is only ROI1.5')
        return value
    expected={'SEALED_USER_STOPPED_ALL8_RESEARCH_TRIAL':('SEALED_USER_STOPPED_ALL8_RESEARCH',False,1.5),
              'SEALED_COMPLETED75_ALL8_RESEARCH_TRIAL':('SEALED_COMPLETED75_ALL8_RESEARCH_NOT_DEPLOYMENT',True,1.15)}
    if status not in expected or not isinstance(seal,dict):raise ValueError('Explicit completed/stopped research seal required')
    seal_status,complete,roi=expected[status]
    if (value!=roi or seal.get('status')!=seal_status or seal.get('formal_training_complete') is not complete or
        seal.get('strict_load') is not True or seal.get('frozen_backbone_verified') is not True or seal.get('roi_expansion')!=roi or
        seal.get('selected_route')!='corner_candidate' or seal.get('deployment_admitted') is not False):raise ValueError('Selected seal/ROI contract differs')
    if complete and seal.get('planned_epochs')!=75:raise ValueError('ROI1.15 requires completed75epoch seal')
    return value
