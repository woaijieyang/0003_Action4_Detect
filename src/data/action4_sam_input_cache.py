"""Exact pinned SAM PNG preprocessing cache; never a resized source-image proxy.

No IO/importing SAM/model initialization at module import. Creation is explicit.
Reader requires caller-held contract and manifest SHA, not cache self-assertion.
"""
from pathlib import Path
import hashlib
import io
import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from data.action4_paths import PROJECT_ROOT, contained_path

SCHEMA = 'action4.sam.cpu_input_tensor_cache.v1'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(value): return hashlib.sha256(value).hexdigest()


def _path(value, exists=True): return contained_path(PROJECT_ROOT, value, exists=exists)


def _stat(path):
    s=path.stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def _versions():
    import torch, torchvision, numpy, PIL
    return {k:v for k,v in [('torch',torch.__version__),('torchvision',torchvision.__version__),('numpy',numpy.__version__),('Pillow',PIL.__version__)]}


def load_pinned_io(source_root):
    """Import the original module, without calling any model builder."""
    import sys, importlib
    root=_path(source_root); expected=root/'sam3/model/io_utils.py'
    _path(expected)
    sys.path.insert(0,str(root))
    module=importlib.import_module('sam3.model.io_utils')
    if Path(module.__file__).resolve()!=expected.resolve():
        raise ValueError('Mixed official SAM IO source imports')
    return module


def make_contract(source_root, region_path, frame_ids):
    """Bind registered frame order and PTS; does not decode/hash original images."""
    root=_path(source_root); region_file=_path(region_path); raw=region_file.read_bytes();region=json.loads(raw)
    source_manifest=region['source_manifest'];mp=_path(source_manifest['path'])
    if _sha(mp.read_bytes())!=source_manifest['sha256']:raise ValueError('Source manifest identity changed')
    ids=list(frame_ids)
    if not ids or any(type(i) is not int or i<0 for i in ids) or len(set(ids))!=len(ids):raise ValueError('Unique ordered actual frame IDs required')
    byid={f['global_frame_id']:f for f in region['frames']}
    rows=[dict(byid[i]) for i in ids]
    if any(type(f['pts']) is not int for f in rows) or any(b['pts']<=a['pts'] for a,b in zip(rows,rows[1:])):raise ValueError('Increasing source PTS required')
    for f in rows: _path(f['image'])
    if any(type(region[k]) is not int or region[k]<=0 for k in ('height','width')):raise ValueError('Native dimensions required')
    return {'schema':SCHEMA,'official_source_root':str(root.relative_to(PROJECT_ROOT)),
        'official_io_sha256':_sha((root/'sam3/model/io_utils.py').read_bytes()),
        'cache_code_sha256':_sha(Path(__file__).read_bytes()),'versions':_versions(),
        'region':{'path':str(region_file.relative_to(PROJECT_ROOT)),'sha256':_sha(raw)},
        'source_manifest':source_manifest,'time_base':region['time_base'],'pixel_space_id':region['pixel_space_id'],
        'native_hw':[region['height'],region['width']],'frames':rows,
        'preprocess':{'image_size':1008,'layout':'CHW','dtype':'float16','mean':[.5,.5,.5],'std':[.5,.5,.5],
        'operation':'official _load_img_as_tensor; assign float32 result to float16 buffer; subtract float16 mean; divide float16 std',
        'resize':'exact pinned torchvision TF.resize(PIL_RGB,size=(1008,1008)) default arguments',
        'offload_video_to_cpu':True}}


def _check_contract(contract):
    if contract.get('schema')!=SCHEMA:raise ValueError('Cache schema mismatch')
    root=_path(contract['official_source_root']);io_path=_path(root/'sam3/model/io_utils.py')
    if _sha(io_path.read_bytes())!=contract['official_io_sha256'] or _sha(Path(__file__).read_bytes())!=contract['cache_code_sha256']:
        raise ValueError('Preprocessor code identity changed')
    if _versions()!=contract['versions']:raise ValueError('Preprocessing library versions changed')
    expected=make_contract(root,contract['region']['path'],[f['global_frame_id'] for f in contract['frames']])
    if expected!=contract:raise ValueError('Stale/foreign preprocessing or frame contract')


def build_cache(directory, contract, *, workers=1):
    """Explicit bounded producer. No hardlinks or changes to original file metadata."""
    import numpy as np, torch
    contract=json.loads(_json(contract))
    _check_contract(contract)
    if type(workers) is not int or not 1<=workers<=min(12,len(os.sched_getaffinity(0))):raise ValueError('Bounded worker count exceeds CPU allocation')
    if torch.get_num_threads()!=1:raise ValueError('Explicit one-thread torch preprocessing required')
    target=_path(directory,exists=False)
    if target.exists():raise FileExistsError('Cache destination must be new; partial failures are retained')
    target.mkdir(parents=True);module=load_pinned_io(contract['official_source_root'])
    def produce(row):
        source=_path(row['image']);before=_stat(source)
        with source.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=row['sha256']:raise ValueError('Original frame SHA mismatch')
        image,h,w=module._load_img_as_tensor(str(source),1008)
        if [h,w]!=contract['native_hw'] or _stat(source)!=before:raise ValueError('Original dimensions/stat changed while decoding')
        if image.dtype!=torch.float32 or image.device.type!='cpu' or tuple(image.shape)!=(3,1008,1008):raise ValueError('Official tensor contract changed')
        tensor=torch.zeros((3,1008,1008),dtype=torch.float16)
        tensor[:]=image
        tensor-=torch.tensor(contract['preprocess']['mean'],dtype=torch.float16)[:,None,None]
        tensor/=torch.tensor(contract['preprocess']['std'],dtype=torch.float16)[:,None,None]
        if not torch.isfinite(tensor).all():raise ValueError('Invalid official-preprocessed tensor')
        return tensor.numpy(),before
    entries=[]
    def commit(index,row,value):
        array,stat=value;name=f'frame_{index:06d}.npy';path=target/name
        with path.open('xb') as f:np.save(f,array,allow_pickle=False)
        data=path.read_bytes()
        entries.append({'file':name,'sha256':_sha(data),'bytes':len(data),'source_stat':stat,'frame':row})
    # At most workers tensors are pending; no unbounded executor.map prefetch.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=deque();iterator=iter(enumerate(contract['frames']))
        for _ in range(workers):
            pair=next(iterator,None)
            if pair is not None:pending.append((*pair,pool.submit(produce,pair[1])))
        while pending:
            index,row,future=pending.popleft();commit(index,row,future.result())
            pair=next(iterator,None)
            if pair is not None:pending.append((*pair,pool.submit(produce,pair[1])))
    manifest={'schema':SCHEMA,'contract':contract,'entries':entries,'complete':True}
    content=_json(manifest);(target/'CACHE.json').write_bytes(content)
    return {'path':str((target/'CACHE.json').relative_to(PROJECT_ROOT)),'sha256':_sha(content),'frames':len(entries)}


class SamInputTensorCache:
    """Read-only tensor source with explicit original video geometry.

    Every fetched payload is checked against a caller-trusted manifest digest.
    Returned tensors own their memory: downstream mutation cannot modify files.
    """
    def __init__(self,manifest_path,*,expected_sha256,expected_contract):
        _check_contract(expected_contract)
        self._manifest_path=_path(manifest_path);self._manifest_stat=_stat(self._manifest_path)
        data=self._manifest_path.read_bytes()
        if _sha(data)!=expected_sha256:raise ValueError('Corrupt/foreign manifest')
        manifest=json.loads(data)
        if manifest.get('schema')!=SCHEMA or manifest.get('complete') is not True or manifest.get('contract')!=expected_contract:raise ValueError('Incomplete/stale/foreign cache')
        entries=manifest['entries']
        if len(entries)!=len(expected_contract['frames']):raise ValueError('Truncated cache frame list')
        for i,(entry,frame) in enumerate(zip(entries,expected_contract['frames'])):
            if entry['frame']!=frame or entry['file']!=f'frame_{i:06d}.npy':raise ValueError('Cache ordering/path changed')
            path=_path(self._manifest_path.parent/entry['file'])
            if path.parent!=self._manifest_path.parent or path.stat().st_size!=entry['bytes']:raise ValueError('Foreign/truncated tensor payload')
            if _stat(_path(frame['image']))!=entry['source_stat']:raise ValueError('Original source changed since cache creation')
        # Private serialized copy prevents caller mutation of admission metadata.
        self._entry_json=tuple(_json(entry) for entry in entries)
        self.video_height,self.video_width=expected_contract['native_hw']
        self.native_hw=(self.video_height,self.video_width)
        self.contract_id=_sha(_json(expected_contract))

    def __len__(self):return len(self._entry_json)

    def __getitem__(self,index):
        import numpy as np,torch
        if type(index) is not int or not 0<=index<len(self):raise IndexError('Explicit cache-local frame index required')
        if _stat(self._manifest_path)!=self._manifest_stat:raise ValueError('Manifest changed after admission')
        entry=json.loads(self._entry_json[index])
        if _stat(_path(entry['frame']['image']))!=entry['source_stat']:raise ValueError('Original source changed after admission')
        path=_path(self._manifest_path.parent/entry['file'])
        if path.parent!=self._manifest_path.parent:raise ValueError('Foreign tensor path after admission')
        payload=path.read_bytes()
        if len(payload)!=entry['bytes'] or _sha(payload)!=entry['sha256']:raise ValueError('Tensor cache corruption/truncation')
        array=np.load(io.BytesIO(payload),allow_pickle=False)
        if array.dtype!=np.float16 or array.shape!=(3,1008,1008) or not np.isfinite(array).all():raise ValueError('Cached tensor dtype/shape/value mismatch')
        return torch.from_numpy(array.copy())

    def as_video_frames(self):
        return self,self.video_height,self.video_width
