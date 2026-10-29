"""Opt-in sequential cached-RGB/native-SAM orientation evidence, no model inference."""
from fractions import Fraction
import hashlib
import threading
import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from model.action4_image_orientation_evidence import ImageOrientationEvidence,DEFAULT_POLICY

ROOT=Path('/data/zy/vln/a-test')


class VideoImageOrientationObserver:
    """Call observe in frame order; one instance owns one nonshared ReceiptVerifier.

    observe(frame_id=..., pts=..., RGB_sha256=..., track_ids=(...)) returns a
    per-track map. Caller may additionally provide time_base for strict checking.
    Evidence only permits retaining the previous filtered R; it estimates no R.
    """
    def __init__(self,input_lock_ref,*,enabled=False,policy=None,track_ids=('track_00','track_01')):
        if type(enabled) is not bool:raise ValueError('Explicit opt-in flag required')
        self.enabled=enabled;self.engine=ImageOrientationEvidence(track_ids,policy=policy)
        self.policy_id=self.engine.policy_id;self.track_ids=self.engine.track_ids
        self._lock=threading.Lock();self.provider=None;self.input_lock_ref=input_lock_ref
        if not enabled:return
        import cv2
        cv2.setNumThreads(1)
        from eval.action4_roi_sweep import read_binding,binding
        from eval.action4_reanchor_roi_provider import ReanchorROIProvider
        ref=binding(input_lock_ref) if isinstance(input_lock_ref,(str,Path)) else input_lock_ref
        lock=read_binding(ref);self.input_lock_ref=ref;self.frames_ref=lock['video']['frames_manifest']
        self.manifest=read_binding(self.frames_ref);self.frames=self.manifest['frames']
        self.time_base=str(Fraction(self.manifest['source']['time_base']))
        self.shape=(self.manifest['source']['height'],self.manifest['source']['width'])
        self.provider=ReanchorROIProvider(**lock['video']['track_replay'],register_project_bodies=True)
        self.replay_ref=lock['video']['track_replay']['predictions_ref']

    def cache_stats(self):return self.engine.cache_stats()

    def observe(self,*,frame_id,pts,RGB_sha256,track_ids=None,time_base=None):
        requested=self.track_ids if track_ids is None else tuple(track_ids)
        if len(set(requested))!=len(requested) or not set(requested)<=set(self.track_ids):raise ValueError('Unknown/duplicate requested track identity')
        if not self.enabled:
            return {tid:dict(track_id=tid,frame_id=frame_id,pts=pts,time_base=time_base,current_RGB_sha256=RGB_sha256,
                current_mask_authenticated=False,available=False,hold_rotation=False,reason='disabled_by_default',
                reference_frame_id=None,previous_frame_id=None,policy_id=self.policy_id,mode='MOTION',stats={},source_refs={}) for tid in requested}
        if not self._lock.acquire(blocking=False):raise RuntimeError('Observer and mutable receipt verifier must be consumed sequentially')
        try:return self._consume(self._prepare(frame_id,pts,RGB_sha256,requested,time_base))
        finally:self._lock.release()

    def _prepare(self,frame_id,pts,RGB_sha256,requested,time_base):
        import cv2
        from data.action4_paths import contained_path
        from data.action4_sam_roi_inputs import array_digest
        if type(frame_id) is not int or not 0<=frame_id<len(self.frames) or type(pts) is not int:raise ValueError('Explicit source frame/PTS required')
        frame=self.frames[frame_id]
        if (frame['frame_id']!=frame_id or frame['pts']!=pts or frame['image_sha256']!=RGB_sha256):raise ValueError('Caller frame/PTS/RGB SHA differs from source')
        if time_base is not None:
            tb=str(Fraction(*time_base)) if isinstance(time_base,(list,tuple)) else str(Fraction(time_base))
            if tb!=self.time_base:raise ValueError('Caller timebase differs from source')
        masks,receipt=self.provider.current_native_masks(frame_id)
        rtb=receipt.get('time_base');rtb=str(Fraction(*rtb)) if isinstance(rtb,(list,tuple)) else str(Fraction(rtb))
        if (receipt['frame_id']!=frame_id or receipt['pts']!=pts or receipt['RGB_sha256']!=RGB_sha256 or rtb!=self.time_base or
            receipt['pixel_space_id']!=self.engine.policy['pixel_space_id']):raise ValueError('Current SAM receipt RGB/frame/PTS/pixel identity differs')
        path=contained_path(ROOT,frame['image'],exists=True);raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=RGB_sha256:raise ValueError('Actual RGB file bytes changed')
        gray=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape!=self.shape or gray.dtype!=np.uint8:raise ValueError('Native grayscale decode failed or changed shape')
        for tid,mask in masks.items():
            if mask.dtype!=np.bool_ or mask.shape!=gray.shape or receipt.get('mask_sha256',{}).get(tid)!=array_digest(mask):
                raise ValueError('Actual mask identity differs from authenticated receipt')
        inputs={}
        for tid in requested:
            body=masks.get(tid);other=np.zeros(gray.shape,bool)
            for oid,mask in masks.items():
                if oid!=tid:other|=mask
            authenticated=(receipt.get('status')=='current_native_response' and receipt.get('per_track_status',{}).get(tid)=='current_mask_present' and body is not None)
            refs=dict(input_lock=self.input_lock_ref,frames_manifest=self.frames_ref,replay=self.replay_ref,
                RGB={'path':frame['image'],'sha256':RGB_sha256},mask_receipt=receipt,
                current_track_mask_sha256=receipt.get('mask_sha256',{}).get(tid))
            inputs[tid]=dict(track_id=tid,frame_id=frame_id,pts=pts,time_base=self.time_base,RGB_sha256=RGB_sha256,
                pixel_space_id=receipt['pixel_space_id'],session_id=receipt.get('session_id'),generation=receipt.get('generation'),
                current_mask_authenticated=authenticated,body_mask=body,other_mask=other,source_refs=refs)
        return dict(gray=gray,frame_id=frame_id,pts=pts,RGB_sha256=RGB_sha256,requested=requested,inputs=inputs)

    def _consume(self,item):
        observations=self.engine.observe(item['gray'],frame_id=item['frame_id'],pts=item['pts'],time_base=self.time_base,
            RGB_sha256=item['RGB_sha256'],track_inputs=item['inputs'])
        return {tid:observations[tid] for tid in item['requested']}

    def iter_observe(self,specifications,*,workers=0):
        """Read ahead only; state transitions remain in caller order and on one thread."""
        if type(workers) is not int or not 0<=workers<=8:raise ValueError('Bounded0..8 image readers required')
        if not workers or not self.enabled:
            for spec in specifications:yield self.observe(**spec)
            return
        if not self._lock.acquire(blocking=False):raise RuntimeError('Concurrent evidence consumption')
        local=threading.local();pending=deque();source=iter(specifications)
        def prepare(spec):
            if not hasattr(local,'reader'):
                from eval.action4_sam_full_review import ReceiptVerifier
                local.reader=copy.copy(self);local.reader.provider=copy.copy(self.provider)
                local.reader.provider.verifier=ReceiptVerifier()
            requested=tuple(spec.get('track_ids',self.track_ids))
            if len(set(requested))!=len(requested) or not set(requested)<=set(self.track_ids):raise ValueError('Unknown/duplicate requested track identity')
            return local.reader._prepare(spec['frame_id'],spec['pts'],spec['RGB_sha256'],requested,spec.get('time_base'))
        try:
            with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='action4-image-evidence') as pool:
                def enqueue():
                    try:spec=next(source)
                    except StopIteration:return False
                    pending.append(pool.submit(prepare,spec));return True
                for _ in range(workers):
                    if not enqueue():break
                try:
                    while pending:
                        item=pending.popleft().result();enqueue()
                        yield self._consume(item)
                finally:
                    for future in pending:future.cancel()
        finally:self._lock.release()