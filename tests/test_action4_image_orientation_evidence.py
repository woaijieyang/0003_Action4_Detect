from unittest.mock import patch
import numpy as np
from model.action4_image_orientation_evidence import ImageOrientationEvidence,body_boundary,boundary_stationarity,_Frame,DEFAULT_POLICY
from eval.action4_video_image_orientation import VideoImageOrientationObserver

SHA='a'*64
MASK=np.zeros((256,256),bool);MASK[30:226,30:226]=True


def inputs(f,tracks=('track_00',),session='s',generation=0):
    return {tid:dict(track_id=tid,frame_id=f,pts=f*400,time_base='1/11988',RGB_sha256=SHA,
        pixel_space_id='raw_video_native_integer_centers',session_id=session,generation=generation,
        current_mask_authenticated=True,body_mask=MASK,other_mask=np.zeros_like(MASK),source_refs={'frame_id':f}) for tid in tracks}


def feed(engine,f,code=0,**kwargs):
    gray=np.zeros(MASK.shape,np.uint8);gray[0,0]=code
    return engine.observe(gray,frame_id=f,pts=f*400,time_base='1/11988',RGB_sha256=SHA,track_inputs=kwargs.get('track_inputs',inputs(f,engine.track_ids)))


def measured_motion(a,b,*args,**kwargs):
    pp=np.array([[x,y] for y in [50.,100.,150.,200.] for x in [50.,100.,150.,200.]])
    theta=np.deg2rad((int(b[0,0])-int(a[0,0]))*.3);r=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    qq=(pp-128)@r.T+128
    return dict(available=True,reason='test_array_correspondences',previous_uv=pp.tolist(),current_uv=qq.tolist(),
        affine_patch=np.c_[r,np.zeros(2)].tolist(),patch_native=[30,30,226,226],patch_size=[196,196],coverage=2.,points=16,inlier_ratio=1.)


def test_five_quiet_frames_then_fixed_hold_reference():
    e=ImageOrientationEvidence(('track_00',))
    with patch('model.action4_image_orientation_evidence.measure_motion',measured_motion):
        rows=[feed(e,f)['track_00'] for f in range(15)]
    assert all(not r['hold_rotation'] for r in rows[:6])
    assert rows[5]['mode']=='HOLD' and rows[5]['reference_frame_id']==5
    assert all(r['hold_rotation'] and r['reference_frame_id']==5 for r in rows[6:])
    assert all(r['stats'].get('body_hull_coverage',[0])[0]<1 for r in rows[1:])


def test_slow_cumulative_rotation_releases_fixed_reference():
    e=ImageOrientationEvidence(('track_00',))
    with patch('model.action4_image_orientation_evidence.measure_motion',measured_motion):
        rows=[feed(e,f,code=f)['track_00'] for f in range(25)]
    releases=[i for i,r in enumerate(rows) if i>5 and rows[i-1]['hold_rotation'] and not r['hold_rotation']]
    assert releases and releases[0]<15
    i=releases[0];assert all(not r['hold_rotation'] for r in rows[i:i+6])


def test_motion_and_unknown_release_immediately():
    for unknown in (False,True):
        e=ImageOrientationEvidence(('track_00',))
        with patch('model.action4_image_orientation_evidence.measure_motion',measured_motion):
            for f in range(7):feed(e,f)
            if unknown:
                with patch('model.action4_image_orientation_evidence.measure_motion',return_value={'available':False,'reason':'no_points'}):r=feed(e,7)['track_00']
                assert not r['available']
            else:r=feed(e,7,code=50)['track_00'];assert r['available']
        assert not r['hold_rotation'] and r['mode']=='MOTION' and e.states['track_00']['reference'] is None


def test_missing_mask_identity_gap_reset():
    for case in ['missing','unauthenticated','cross_track','session','generation','gap','timebase']:
        e=ImageOrientationEvidence(('track_00',))
        with patch('model.action4_image_orientation_evidence.measure_motion',measured_motion):
            for f in range(7):feed(e,f)
            f=9 if case=='gap' else 7;d=inputs(f)
            if case=='missing':d['track_00']['body_mask']=None
            if case=='unauthenticated':d['track_00']['current_mask_authenticated']=False
            if case=='cross_track':d['track_00']['track_id']='track_01'
            if case=='session':d['track_00']['session_id']='new'
            if case=='generation':d['track_00']['generation']=1
            if case=='timebase':d['track_00']['time_base']='1/1000'
            r=feed(e,f,track_inputs=d)['track_00']
        assert not r['hold_rotation'] and not r['available'] and r['mode']=='MOTION',case
        assert e.states['track_00']['reference'] is None


def test_cache_bounded_and_shared_previous_gray():
    e=ImageOrientationEvidence(('track_00','track_01'))
    with patch('model.action4_image_orientation_evidence.measure_motion',measured_motion):
        for f in range(120):
            d=inputs(f,e.track_ids)
            if f%17==0:d['track_01']['current_mask_authenticated']=False
            feed(e,f,track_inputs=d);stats=e.cache_stats()
            assert stats['retained_gray_images']<=3 and stats['reference_slots']<=2
            if len(e.previous)==2:assert e.previous['track_00'].gray is e.previous['track_01'].gray
    e.clear();assert e.cache_stats()['retained_gray_images']==0


def test_native_real_optical_flow_static_translation_and_rotation():
    import cv2
    rng=np.random.default_rng(13);gray=rng.integers(0,256,MASK.shape,dtype=np.uint8)
    def frame(image,mask,f):
        band,box=body_boundary(mask,np.zeros_like(mask));return _Frame(image,mask,band,box,f,f*400,'1/11988',SHA,('test',),{})
    a=frame(gray,MASK,0);b=frame(gray.copy(),MASK,1)
    available,quiet,_,stats=boundary_stationarity(a,b);assert available and quiet
    move=np.array([[1.,0,5],[0,1.,3]])
    g=cv2.warpAffine(gray,move,(256,256));m=cv2.warpAffine(MASK.astype(np.uint8),move,(256,256),flags=cv2.INTER_NEAREST)>0
    available,quiet,_,stats=boundary_stationarity(a,frame(g,m,2));assert available and quiet
    rotate=cv2.getRotationMatrix2D((128,128),10,1);g=cv2.warpAffine(gray,rotate,(256,256));m=cv2.warpAffine(MASK.astype(np.uint8),rotate,(256,256),flags=cv2.INTER_NEAREST)>0
    available,quiet,_,stats=boundary_stationarity(a,frame(g,m,3));assert not (available and quiet)


def test_ring_excludes_edge_core_and_other():
    other=np.zeros_like(MASK);other[:,50:60]=True;band,box=body_boundary(MASK,other)
    assert not band[30:33].any() and not band[128,128] and not band[:,50:60].any()
    assert band[40,100]


def test_default_off_does_no_io():
    observer=VideoImageOrientationObserver('this_missing_path_must_not_be_read')
    r=observer.observe(frame_id=4,pts=1600,RGB_sha256=SHA,track_ids=('track_00',))['track_00']
    assert not r['available'] and not r['hold_rotation'] and observer.provider is None

def test_eval_caller_and_receipt_identity_fail_before_RGB_IO():
    observer=VideoImageOrientationObserver('unused',enabled=False)
    observer.enabled=True;observer.time_base='1/11988';observer.shape=(256,256)
    observer.frames=[{'frame_id':0,'pts':0,'image_sha256':SHA,'image':'must_not_be_read.png'}]
    class Provider:
        calls=0
        def current_native_masks(self,f):
            self.calls+=1
            return {},dict(frame_id=f,pts=999,RGB_sha256=SHA,time_base=[1,11988],pixel_space_id='raw_video_native_integer_centers')
    observer.provider=Provider()
    for pts,sha in [(1,SHA),(0,'b'*64)]:
        try:observer.observe(frame_id=0,pts=pts,RGB_sha256=sha,track_ids=('track_00',))
        except ValueError:pass
        else:raise AssertionError('caller identity accepted')
    assert observer.provider.calls==0
    try:observer.observe(frame_id=0,pts=0,RGB_sha256=SHA,track_ids=('track_00',))
    except ValueError:pass
    else:raise AssertionError('receipt identity accepted')
    assert observer.provider.calls==1 and observer.cache_stats()['retained_gray_images']==0
