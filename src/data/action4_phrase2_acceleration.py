"""Explicit new-worker-only Cycles session retention and assigned-device cache.

Rendering, projection, masks and surface algorithms remain the frozen modules.
Only GPU/session setup is reused. Every returned record binds this implementation.
"""
from pathlib import Path
import os,copy,hashlib
R=Path('/data/zy/vln/a-test')

def install(entrypoint):
    from data import action4_phrase2_runtime as runtime
    from data import action4_synthesis as synth
    from data import action4_phrase2_pixels as pixels
    import bpy
    paths=[Path(__file__).resolve(),Path(entrypoint).resolve()]
    if any(not p.is_relative_to(R) for p in paths):raise ValueError('Acceleration implementation escapes project')
    sources={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    original_class=runtime.Phrase2Runtime;original_device=synth._render_device;original_frame=pixels.render_frame
    state={'devices':{},'cache_hits':0}
    class PersistentRuntime(original_class):
        def __init__(self,name):
            super().__init__(name)
            self.scene.render.use_persistent_data=True
            self.scene.cycles.device='GPU'
    def cached_device(scene,settings):
        key=(settings['device'],settings.get('worker_gpu_uuid'))
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=key[1]:raise ValueError('Assigned GPU environment changed')
        if key not in state['devices']:
            state['devices'][key]=original_device(scene,settings)
        else:
            prefs=bpy.context.preferences.addons['cycles'].preferences
            expected={d['id'] for d in state['devices'][key]['devices']}
            actual={d.id for d in prefs.devices if d.type==key[0] and d.use}
            if prefs.compute_device_type!=key[0] or actual!=expected:raise ValueError('Frozen GPU preferences changed')
            scene.cycles.device='GPU';state['cache_hits']+=1
        return copy.deepcopy(state['devices'][key])
    def render_frame(**kwargs):
        if not kwargs['scene'].render.use_persistent_data:raise ValueError('Persistent profile changed')
        ids=dict(kwargs['source_identities']);ids.update(sources);kwargs['source_identities']=ids
        result=original_frame(**kwargs)
        result['record']['acceleration_profile']={'version':'cycles_persistent_assigned_device_cache_v002','sources':sources,'device_initializations':len(state['devices']),'validated_device_cache_hits':state['cache_hits'],'geometry_or_sampling_changed':False}
        result['record']['render_settings'].update(persistent_data=True,assigned_device_initialization='once_then_preferences_and_environment_checked_each_frame')
        return result
    runtime.Phrase2Runtime=PersistentRuntime;synth._render_device=cached_device;pixels.render_frame=render_frame
    return state
