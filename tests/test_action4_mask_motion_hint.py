import copy
import numpy as np
from model.action4_mask_motion_hint import DEFAULT_POLICY,evaluate_mask_motion_hint,measure_mask_motion,mask_digest,scalar_pass


def fixture():
    mask=np.zeros((240,320),bool);mask[40:200,50:270]=True
    def identity(f):return dict(track_id='track_00',frame_id=f,pts=f*400,time_base=[1,11988],RGB_sha256='a'*64,pixel_space_id='raw_video_native_integer_centers',mask_authenticated=True,session_id='session',generation=0,mask_sha256=mask_digest(mask),source_receipt={'test_fixture':True,'frame_id':f})
    e=dict(track_id='track_00',frame_id=11,pts=4400,time_base='1/11988',current_RGB_sha256='a'*64,available=False,hold_rotation=False,current_mask_authenticated=True,policy_id='boundary')
    return dict(previous_mask=mask,current_mask=mask.copy(),track_id='track_00',previous_identity=identity(10),current_identity=identity(11),boundary_evidence=e)


def test_static_unknown_only_weak_gain():
    k=fixture();r=evaluate_mask_motion_hint(**k)
    assert r['weak_low_motion'] and r['eligible_for_gain'] and r['scalar_available']
    assert not any(key in r for key in ('available','hold_rotation','pose_accepted','stationarity'))
    assert r['frame_id']==11 and r['previous_frame_id']==10 and r['current_mask_authenticated']


def test_strong_boundary_never_overridden():
    for hold in [False,True]:
        k=fixture();k['boundary_evidence'].update(available=True,hold_rotation=hold)
        r=evaluate_mask_motion_hint(**k);assert not r['eligible_for_gain'] and not r['weak_low_motion'] and r['scalar_thresholds_pass']
    k=fixture();k['boundary_evidence']['hold_rotation']=True
    assert not evaluate_mask_motion_hint(**k)['eligible_for_gain']


def test_missing_empty_and_identity_changes_fail_closed():
    for name in ['track_id','frame_id','pts','time_base','session_id','generation','mask_authenticated','mask_sha256']:
        k=fixture()
        changes={'track_id':'other','frame_id':13,'pts':4000,'time_base':'1/1000','session_id':'different','generation':1,'mask_authenticated':False,'mask_sha256':'b'*64}
        k['current_identity'][name]=changes[name]
        r=evaluate_mask_motion_hint(**k);assert not r['weak_low_motion'] and not r['eligible_for_gain'],name
    for mask in [None,np.zeros((240,320),bool),np.ones((240,320),np.uint8)]:
        k=fixture();k['current_mask']=mask;r=evaluate_mask_motion_hint(**k);assert not r['eligible_for_gain'] and not r['weak_low_motion']


def test_saved_boundary_identity_mismatch_rejected():
    for field,value in [('frame_id',12),('track_id','track_01'),('pts',4401),('time_base','1/12000'),('current_RGB_sha256','b'*64)]:
        k=fixture();k['boundary_evidence'][field]=value;r=evaluate_mask_motion_hint(**k);assert not r['eligible_for_gain'] and not r['weak_low_motion']


def test_native_metrics_match_original_nonzero_recipe():
    import cv2
    a=fixture()['previous_mask'];b=cv2.warpAffine(a.astype(np.uint8),np.array([[1.,0,2],[0,1.,-1]]),(320,240),flags=cv2.INTER_NEAREST)>0
    s=measure_mask_motion(a,b);ay,ax=np.nonzero(a);by,bx=np.nonzero(b);delta=np.array([bx.mean()-ax.mean(),by.mean()-ay.mean()]);diag=min(np.hypot(np.ptp(ax)+1,np.ptp(ay)+1),np.hypot(np.ptp(bx)+1,np.ptp(by)+1))
    assert np.array_equal(s['centroid_translation_native_px'],delta)
    assert s['centroid_displacement_fraction']==float(np.linalg.norm(delta)/diag) and s['warp_nearest_IoU']==1 and s['absolute_log_area_ratio']==0
    assert scalar_pass(s)


def test_motion_scale_shape_and_exact_thresholds():
    import cv2
    a=fixture()['previous_mask'];b=cv2.warpAffine(a.astype(np.uint8),np.array([[1.,0,10],[0,1.,0]]),(320,240),flags=cv2.INTER_NEAREST)>0
    assert not scalar_pass(measure_mask_motion(a,b))
    b=a.copy();b[40:60]=False;assert not scalar_pass(measure_mask_motion(a,b))
    edge=dict(centroid_displacement_fraction=.015,warp_nearest_IoU=.95,absolute_log_area_ratio=.03);assert scalar_pass(edge)
    for field,value in [('centroid_displacement_fraction',.01500001),('warp_nearest_IoU',.9499999),('absolute_log_area_ratio',.0300001)]:
        d=dict(edge);d[field]=value;assert not scalar_pass(d)


def test_readonly_inputs_and_canonical_rational():
    k=fixture();a=k['previous_mask'].copy();b=k['current_mask'].copy();e=copy.deepcopy(k['boundary_evidence']);r=evaluate_mask_motion_hint(**k)
    assert np.array_equal(a,k['previous_mask']) and np.array_equal(b,k['current_mask']) and e==k['boundary_evidence']
    assert r['time_base']=='1/11988'


def test_first_frame_retains_current_identity_unknown():
    k=fixture();k['previous_identity']=None;k['previous_mask']=None;r=evaluate_mask_motion_hint(**k)
    assert r['frame_id']==11 and r['pts']==4400 and r['current_RGB_sha256']=='a'*64 and r['current_mask_authenticated']
    assert not r['weak_low_motion'] and not r['eligible_for_gain'] and r['previous_frame_id'] is None