import copy
import numpy as np
from model.action4_native_keyframe_policy import DEFAULT_POLICY,decide_keyframe
from eval.action4_native_clip import keyframe_host_options,native_pose_key


def fixture():
    uv=np.array([[x,y] for y in [190.,240.,290.] for x in [260.,300.,340.,380.]])
    obs=np.zeros((12,10));obs[:,:2]=uv;obs[:,5:7]=1;obs[:,7:9]=(uv-[320,240])/1000
    pose=np.eye(4);pose[2,3]=1
    cad=np.array([[x,y,z] for x in [-.1,.1] for y in [-.1,.1] for z in [-.05,.05]])
    mask=np.zeros((480,640),np.uint8);mask[160:320,240:400]=255
    return dict(policy=copy.deepcopy(DEFAULT_POLICY),track_id='track_00',frame_id=10,pending_frame=10,pending_support_frame=10,
        native_ok=True,pose=pose,before_pose=pose.copy(),canonical=cad,K=np.array([[1000.,0,320],[0,1000,240],[0,0,1]]),
        observations=obs,owned_mask=mask,body_mask=mask.copy(),support_audit={'frame_id':10,'track_id':'track_00','status':'current_support','native_mask_sha256':'abc'},
        expected_mask_sha256='abc',current_sam=True,predicted_box=[200.,100.,440.,380.])


def test_valid_engineering_observations_only():
    d=decide_keyframe(**fixture());assert d['commit_recommended'] and d['unique_observations']==12 and d['grid_cells']>=4
    assert not d['geometry_accepted'] and not d['pose_accuracy_verified'] and d['body_coverage']==1


def test_duplicate_uv_cannot_manufacture_twelve_points():
    k=fixture();k['observations']=np.repeat(k['observations'][:3],10,axis=0)
    d=decide_keyframe(**k);assert not d['commit_recommended'] and d['unique_observations']==3


def test_final_pose_residual_not_stale_linearization():
    k=fixture();k['pose'][0,3]=.01
    d=decide_keyframe(**k);assert not d['commit_recommended'] and d['residual_median_px']>9.99 and 'reprojection_residual' in d['failed_gates']


def test_duplicate_pixel_uses_worst_residual():
    k=fixture();bad=k['observations'].copy();bad[:,7]+=.02;k['observations']=np.concatenate([k['observations'],bad])
    d=decide_keyframe(**k);assert d['unique_observations']==12 and d['residual_median_px']>19.99 and not d['commit_recommended']


def test_unknown_identity_or_failed_native_rejected():
    for name,value in [('pending_frame',11),('pending_support_frame',11),('native_ok',False),('current_sam',None),('expected_mask_sha256','other')]:
        k=fixture();k[name]=value;assert not decide_keyframe(**k)['commit_recommended'],name
    k=fixture();k['support_audit']['track_id']='track_01';assert not decide_keyframe(**k)['commit_recommended']


def test_geometry_and_body_coverage_guards():
    k=fixture();k['pose'][0,0]=2;assert decide_keyframe(**k)['reason']=='nonrigid_pose'
    k=fixture();k['pose'][2,3]=-.1;assert decide_keyframe(**k)['reason']=='nonpositive_canonical_depth'
    k=fixture();k['body_mask'][:]=255;d=decide_keyframe(**k);assert not d['commit_recommended'] and 'body_coverage' in d['failed_gates']
    k=fixture();k['pose'][0,3]=.2;d=decide_keyframe(**k);assert 'translation_change' in d['failed_gates']
    k=fixture();theta=np.deg2rad(40);k['pose'][:3,:3]=[[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]
    d=decide_keyframe(**k);assert 'rotation_change' in d['failed_gates']


def test_grid_and_current_owned_support_not_just_count():
    k=fixture();uv=np.array([[261+i*.2,191.] for i in range(12)]);k['observations'][:,:2]=uv;k['observations'][:,7:9]=(uv-[320,240])/1000
    d=decide_keyframe(**k);assert not d['commit_recommended'] and d['grid_cells']==1
    k=fixture();k['owned_mask'][:]=0;assert decide_keyframe(**k)['reason']=='empty_current_support'


def test_defaults_and_selected_pose_route_preserved():
    assert keyframe_host_options({})==(False,'current_evidence_optional_reseed_and_SAM_only_continuation',None)
    assert native_pose_key({})=='raw_A0_pose'
    assert native_pose_key({'native_pose_input_kind':'selected_A0'})=='selected_A0_pose'
    assert keyframe_host_options({'defer_keyframe_commit':True,'keyframe_host_policy':DEFAULT_POLICY,'recovery_mode':'record_only_no_reseed'})[0]
    try:keyframe_host_options({'defer_keyframe_commit':True})
    except ValueError:pass
    else:raise AssertionError('missing policy accepted')