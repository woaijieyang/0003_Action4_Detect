"""Persistent wearing template for the explicit fast execution path.
A lease restores to its wearing template; owner closes to original base once.
"""
import os,threading,time
from copy import deepcopy
from . import action4_modular_scene_sampling as sampling
from . import action4_modular_wearing_base as wearing
from . import action4_modular_pose_controls as pose
from . import action4_modular_assembly as assembly
from . import action4_modular_render as neutral

class PoseLease:
    def __init__(self,owner,applied):self.owner=owner;self.applied=applied;self.restored=False
    def __getattr__(self,name):return getattr(self.applied,name)
    def restore(self):
        self.owner.guard()
        if self.restored or self.owner.active is not self:raise ValueError('Exact active lease required')
        self.applied._restore_to_base()
        self.applied.restored=True;self.restored=True;self.owner.active=None
        return {'status':'RESTORED_PERSISTENT_WEARING_TEMPLATE','original_source_restored':False}
    def __enter__(self):return self
    def __exit__(self,kind,error,tb):
        try:self.restore()
        finally:
            if error is not None:self.owner.poisoned=True
        return False

class PersistentWearingBase:
    def __init__(self,context):
        self.pid=os.getpid();self.thread=threading.get_ident();self.context=context;self.closed=False;self.poisoned=False;self.active=None
        start=time.perf_counter();self.base=wearing.create(context['wearing_context']);self.base_create_seconds=time.perf_counter()-start
        try:
            self.scene=self.base.scene;self.bindings=sampling.actor_bindings(self.scene,context);self.baseline=sampling._snapshot(self.scene);self.apply_count=0
        except BaseException:
            self.poisoned=True
            try:self.base.restore()
            finally:self.closed=True
            raise
    def guard(self):
        if self.closed or self.poisoned or self.pid!=os.getpid() or self.thread!=threading.get_ident():raise ValueError('Same-owner healthy persistent base required')
    def apply(self,parameters):
        self.guard()
        if self.active is not None:raise ValueError('Concurrent pose leases prohibited')
        p=pose.validate_pose_controls(parameters,self.context);self.base.verifier(self.scene)
        expected=sampling.expected_state(self.baseline,self.bindings,p,self.context['source_context'])
        applied=sampling.AppliedScene(self.scene,self.context,self.base.contract,p,self.bindings,self.baseline,expected,base_session=self.base)
        lease=PoseLease(self,applied);self.active=lease
        try:
            for side in ('left','right'):
                for name in self.bindings[side]['roots']:assembly.set_rigid_basis(self.scene.objects[name],expected['objects'][name]['basis'])
                for name in self.bindings[side]['names']:
                    row=expected['objects'][name];self.scene.objects[name].hide_render=row['hide_render'];self.scene.objects[name].pass_index=row['pass_index']
            for name in neutral.ENV_NAMES[:2]:self.scene.objects[name].hide_render=expected['objects'][name]['hide_render']
            lamp=neutral.ENV_NAMES[2];self.scene.objects[lamp].data.energy=expected['objects'][lamp]['light']['properties']['energy'];self.scene.view_settings.exposure=expected['view']['exposure'];sampling._update();applied.verifier(self.scene)
            applied.contract['actor_collision_diagnostics']=sampling.geometry_diagnostics(applied);applied.verifier(self.scene)
            applied.contract['pose_controls_execution']={'provenance':deepcopy(p['provenance']),'controls':deepcopy(p['controls']),'formal_entry_adapter':p['formal_entry_adapter'],'training_accepted':False}
            applied.contract['pose_controls_geometry']=pose.verify_actual_pose_geometry(applied);self.apply_count+=1;return lease
        except BaseException:
            try:lease.restore()
            finally:self.poisoned=True
            raise
    def close(self):
        if (self.pid,self.thread)!=(os.getpid(),threading.get_ident()):raise ValueError('Same-owner close required')
        if self.closed:raise ValueError('One owner close only')
        cleanup_error=None
        if self.active is not None:
            try:self.active.restore()
            except BaseException as exc:cleanup_error=exc;self.poisoned=True
        try:receipt=self.base.restore()
        finally:self.closed=True
        if cleanup_error is not None:raise RuntimeError('Lease cleanup failed; original base restoration attempted') from cleanup_error
        if not self.base.restored:raise ValueError('Original base restoration incomplete')
        return {'status':'PERSISTENT_BASE_CLOSED','base_create_count':1,'apply_count':self.apply_count,'base_restoration':receipt,'original_source_restored':True,'poisoned':self.poisoned}
