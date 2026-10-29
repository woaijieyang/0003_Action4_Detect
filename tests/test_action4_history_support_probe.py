from dataclasses import asdict
import numpy as np
from eval.action4_history_support_probe import map_translation,temporal_eligibility,support_check,choose_candidate,SupportPolicy


def test_roi_translation_identity_and_scale_without_extra_expansion():
    k=np.array([[1000.,0,320],[0,1000,240],[0,0,1]]);t=np.array([.02,.01,1.]);a=np.array([100.,200,300,400]);b=np.array([200.,300,600,700])
    same,m=map_translation(t,k,a,a);assert np.allclose(same,t,rtol=0,atol=1e-12) and m['additional_ROI_expansion'] is False
    changed,m=map_translation(t,k,a,b);assert np.allclose(changed,[.18,.08,.5]) and m['mapped_z_m']==.5


def test_temporal_domain_and_native_generation_gate():
    r=dict(session_id='s',generation=0,pixel_space_id='native',time_base=[1,1000])
    assert temporal_eligibility(5,10,0,250,[1,1000],r,r)[0]
    assert not temporal_eligibility(5,10,0,251,[1,1000],r,r)[0]
    assert not temporal_eligibility(11,10,300,250,[1,1000],r,r)[0]
    assert not temporal_eligibility(4,10,0,250,[1,1000],r,r)[0]
    for changed in [dict(r,generation=1),dict(r,session_id='new'),dict(r,time_base=[1,999])]:assert not temporal_eligibility(9,10,200,250,[1,1000],r,changed)[0]


def test_selection_uses_angle_then_newness_not_IoU():
    a=dict(source_frame_id=8,rotation=np.eye(3).tolist(),support={'reference_compatible':True,'IoU_diagnostic_only':.99})
    b=dict(source_frame_id=9,rotation=np.eye(3).tolist(),support={'reference_compatible':True,'IoU_diagnostic_only':.01})
    c=dict(source_frame_id=10,rotation=np.eye(3).tolist(),support={'reference_compatible':False,'IoU_diagnostic_only':1.})
    assert choose_candidate([a,b,c],np.eye(3)) is b
    assert choose_candidate([c],np.eye(3)) is None


def test_support_policy_original_and_exception_retained():
    assert asdict(SupportPolicy())==dict(min_visible_coverage=.9,min_visible_over_projection=.4,tolerance_diagonal_fraction=.01,min_mask_pixels=32)
    cad=np.array([[x,y,z] for x in [-.1,.1] for y in [-.1,.1] for z in [-.1,.1]])
    k=np.array([[100.,0,100],[0,100.,100],[0,0,1.]])
    mask=np.zeros((200,200),bool);mask[92:108,92:108]=True;other=np.zeros_like(mask)
    good=support_check(mask,other,k,cad,np.eye(3),[0,0,1]);assert good['reference_compatible']
    failed=support_check(mask,other,k,cad,np.eye(3),[0,0,.01]);assert not failed['reference_compatible'] and 'Nonpositive' in failed['error']
    another=support_check(mask,other,k,cad,np.eye(3),[0,0,1]);assert another['reference_compatible']
    absent=support_check(None,other,k,cad,np.eye(3),[0,0,1]);assert not absent['reference_compatible']