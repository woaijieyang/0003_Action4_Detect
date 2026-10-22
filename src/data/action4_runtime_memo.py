"""Explicit process-local, owner-checked memo policy; original files untouched.
Geometry reads fresh binary buffers on every call. Input snapshots are limited
to approved immutable plan inputs, guarded by stat and fresh close SHA checks.
Runtime adapters are installed only inside this context and restored on exit.
"""
from pathlib import Path
import hashlib,json,os,threading,time,struct,inspect,functools
import numpy as np
R=Path('/data/zy/vln/a-test')

def bounded(value):
    p=Path(value)
    if '..' in p.parts:raise ValueError('No traversal')
    p=(p if p.is_absolute() else R/p).resolve(strict=True)
    if not p.is_relative_to(R):raise ValueError('Project boundary')
    return p

def signature(p):
    s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

class GeometryDigestCache:
    def __init__(self,reference):
        self.reference=reference;self.owner=(os.getpid(),threading.get_ident());self.cache={};self.calls=0;self.hits=0;self.misses=0
    def __call__(self,obj):
        if self.owner!=(os.getpid(),threading.get_ident()):raise ValueError('Geometry cache owner mismatch')
        mesh=obj.data;h=hashlib.sha256()
        def collect(collection,attribute,width,dtype):
            data=np.empty(len(collection)*width,dtype=dtype);collection.foreach_get(attribute,data)
            h.update(struct.pack('<QQ',len(collection),width));h.update(data.tobytes())
        collect(mesh.vertices,'co',3,np.float32)
        collect(mesh.loops,'vertex_index',1,np.int32)
        collect(mesh.polygons,'loop_start',1,np.int32);collect(mesh.polygons,'loop_total',1,np.int32)
        h.update(struct.pack('<Q',len(mesh.uv_layers)))
        for layer in mesh.uv_layers:
            name=layer.name.encode();h.update(struct.pack('<Q',len(name)));h.update(name);collect(layer.data,'uv',2,np.float32)
        key=mesh.as_pointer();digest=h.digest();self.calls+=1;old=self.cache.get(key)
        if old is not None and old[0]==digest:self.hits+=1;return old[1]
        value=self.reference(obj);self.cache[key]=(digest,value);self.misses+=1;return value

class InputSnapshotCache:
    def __init__(self,allowed,max_bytes=1024**3):
        self.allowed=dict(allowed);self.owner=(os.getpid(),threading.get_ident());self.values={};self.bytes_used=0;self.max_bytes=max_bytes;self.hits=0;self.reads=0
    def guard(self):
        if self.owner!=(os.getpid(),threading.get_ident()):raise ValueError('Input cache owner mismatch')
    def lookup(self,value):
        self.guard()
        p=bounded(value);name=str(p.relative_to(R))
        return p,name,self.allowed.get(name)
    def read(self,value,expected=None):
        p,name,allowed=self.lookup(value)
        if allowed is None:return None
        if expected is not None and expected!=allowed:raise ValueError('Conflicting approved input SHA')
        current=signature(p);saved=self.values.get(name)
        if saved is not None:
            if current!=saved[0]:raise ValueError('Immutable input changed during memo scope: '+name)
            self.hits+=1
            if saved[1] is not None:return saved[1]
        data=p.read_bytes();self.reads+=1
        if signature(p)!=current or hashlib.sha256(data).hexdigest()!=allowed:raise ValueError('Input bytes changed: '+name)
        retained=data if self.bytes_used+len(data)<=self.max_bytes else None
        if saved is None:
            self.values[name]=(current,retained)
            if retained is not None:self.bytes_used+=len(retained)
        return data
    def digest(self,value):
        p,name,allowed=self.lookup(value)
        if allowed is None:return None
        saved=self.values.get(name)
        if saved is not None:
            if signature(p)!=saved[0]:raise ValueError('Immutable input changed during memo scope: '+name)
            self.hits+=1;return allowed
        self.read(p);return allowed
    def close(self):
        self.guard()
        for name,(stat,_) in self.values.items():
            p=bounded(name)
            if signature(p)!=stat or hashlib.sha256(p.read_bytes()).hexdigest()!=self.allowed[name]:raise ValueError('Fresh input closing failed: '+name)
        return {'unique_inputs':len(self.values),'byte_reads':self.reads,'hits':self.hits,'retained_bytes':self.bytes_used,'fresh_close_verified':True}

class RuntimeMemo:
    def __init__(self,allowed):self.inputs=InputSnapshotCache(allowed);self.adapters=[];self.closed=False;self.entered=False;self.receipt=None
    def install(self,obj,name,replacement):
        old=getattr(obj,name);self.adapters.append((obj,name,old,replacement));setattr(obj,name,replacement)
    def __enter__(self):
        self.inputs.guard()
        if self.entered or self.closed:raise ValueError('One memo scope entry only')
        self.entered=True
        try:return self._enter()
        except BaseException:
            for obj,name,old,replacement in reversed(self.adapters):setattr(obj,name,old)
            self.closed=True
            raise
    def _enter(self):
        from . import action4_global_scene as scene,action4_execution_context_bridge as bridge
        from . import action4_synthesis as synth,action4_modular_bilateral_scene as bilateral,action4_modular_render as neutral,action4_modular_assembly as assembly,action4_global_environment as env
        if getattr(scene._geometry,'__module__',None)!='data.action4_global_scene' or getattr(scene._geometry,'__name__',None)!='_geometry':raise ValueError('Geometry provider already adapted')
        for module in (scene,bridge,synth,bilateral,neutral,assembly,env):
            path=bounded(module.__file__);name=str(path.relative_to(R))
            if name not in self.inputs.allowed:raise ValueError('Adapter source absent from approved inputs')
            self.inputs.read(path,self.inputs.allowed[name])
        self.geometry=GeometryDigestCache(scene._geometry);self.install(scene,'_geometry',self.geometry)
        for module,name in [(synth,'_hash'),(bilateral,'sha'),(neutral,'sha'),(assembly,'sha'),(env,'sha')]:
            old=getattr(module,name)
            sig=inspect.signature(old)
            @functools.wraps(old)
            def cached(*args,_old=old,_sig=sig,**kwargs):
                bound=_sig.bind(*args,**kwargs);value=next(iter(bound.arguments.values()))
                value_hash=self.inputs.digest(value)
                return _old(*args,**kwargs) if value_hash is None else value_hash
            self.install(module,name,cached)
        original=bridge.ExecutionBridge.live_bytes
        snapshot=self.inputs
        @functools.wraps(original)
        def live_bytes(self,r):
            bridge._ref_shape(r);data=snapshot.read(r['path'],r['sha256'])
            if data is None:return original(self,r)
            bridge._remember(self.dependencies,r);return data
        self.install(bridge.ExecutionBridge,'live_bytes',live_bytes)
        return self
    def __exit__(self,kind,error,tb):
        self.inputs.guard()
        if not self.entered or self.closed:raise ValueError('Open owner memo scope required')
        drift=[]
        for obj,name,old,replacement in reversed(self.adapters):
            if getattr(obj,name) is not replacement:drift.append(name)
            setattr(obj,name,old)
        self.closed=True
        receipt=self.inputs.close();receipt.update(geometry_calls=self.geometry.calls,geometry_hits=self.geometry.hits,geometry_misses=self.geometry.misses,explicit_runtime_adapters_restored=True)
        if drift:raise RuntimeError('Runtime adapter ownership changed: '+str(drift))
        self.receipt=receipt
        return False
